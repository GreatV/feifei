import os
import sys
import time
import json
import logging
import mimetypes
import concurrent.futures
from github import Github
from git import Repo
from threading import Lock
import shutil
import fcntl
from dotenv import load_dotenv
import yaml

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from a .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Lock for thread-safe file operations
processed_issues_lock = Lock()

# Load configuration from YAML file
def load_config(yaml_file='config.yaml'):
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {yaml_file}")
    except FileNotFoundError:
        logging.error(f"Configuration file {yaml_file} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    return config

# Load sensitive information from environment variables
def load_sensitive_config():
    sensitive_config = {
        "github_token": os.getenv("GITHUB_TOKEN"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "huggingfacehub_api_token": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    }
    missing_keys = [k for k, v in sensitive_config.items() if not v]
    if missing_keys:
        logging.warning(
            f"Missing environment variables for sensitive config: {', '.join(missing_keys)}"
        )
    return sensitive_config

# Global CONFIG dictionary combining YAML config and environment variables
CONFIG = {}
CONFIG.update(load_config())
CONFIG.update(load_sensitive_config())

def initialize_llm(backend="openai", **kwargs):
    """
    Initialize the Language Model (LLM) based on the selected backend.
    """
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-3.5-turbo"),
            openai_api_key=CONFIG.get("openai_api_key"),
            openai_api_base=kwargs.get("openai_api_base"),
        )
    elif backend == "huggingface":
        from langchain_community.llms import HuggingFaceHub

        llm = HuggingFaceHub(
            repo_id=kwargs.get("repo_id", "gpt2"),
            huggingfacehub_api_token=CONFIG.get("huggingfacehub_api_token"),
            api_base=kwargs.get("huggingface_api_base"),
        )
    elif backend == "local":
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_name = kwargs.get("model_name", "your-local-model-name")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            device=kwargs.get("device", -1),
        )

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")
    return llm


def initialize_embeddings(backend="openai", **kwargs):
    """
    Initialize the embeddings model based on the selected backend.
    """
    if backend == "openai":
        embeddings = OpenAIEmbeddings(
            openai_api_key=CONFIG.get("openai_api_key"),
            openai_api_base=kwargs.get("openai_api_base"),
        )
    elif backend == "huggingface" or backend == "local":
        model_name = kwargs.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    return embeddings


def get_prompt_template():
    """
    Retrieve the prompt template.
    """
    prompt_template = """
You are a smart assistant capable of answering user questions based on provided documents.

Question: {input}

Here are some potentially useful documents:
{context}

Please provide a detailed answer based on the above information, in the same language as the question. If possible, reference related Issues or Discussions and provide links.

Answer:
"""
    return PromptTemplate(
        template=prompt_template, input_variables=["input", "context"]
    )


def format_document(doc):
    """
    Format a document to include metadata for the context.
    """
    title = doc.metadata.get("title", "Untitled")
    url = doc.metadata.get("url", "")
    return f"Title: {title}\nLink: {url}\nContent: {doc.page_content}\n"


def load_or_build_vectorstore(repo, embeddings, github_token):
    """
    Load an existing vector store or build a new one if the repository has new commits.
    """
    VECTORSTORE_PATH = os.path.join(
        os.getcwd(), f"{repo.full_name.replace('/', '_')}_vectorstore"
    )
    REPO_STATE_PATH = os.path.join(
        os.getcwd(), f"{repo.full_name.replace('/', '_')}_repo_state.json"
    )

    try:
        # Check if the vector store file exists and is not empty
        if os.path.exists(VECTORSTORE_PATH) and os.path.getsize(VECTORSTORE_PATH) > 0:
            with open(REPO_STATE_PATH, "r") as f:
                repo_state = json.load(f)

            latest_commit = repo.get_commits()[0]
            latest_commit_sha = latest_commit.sha

            if repo_state.get("latest_commit_sha") == latest_commit_sha:
                try:
                    # Attempt to load the FAISS index
                    vectorstore = FAISS.load_local(
                        VECTORSTORE_PATH,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logging.info(f"Loaded existing vector store for {repo.full_name}")
                    return vectorstore
                except Exception as e:
                    logging.error(f"Failed to load FAISS index. Rebuilding: {e}")

        # If the vector store is missing, empty, or fails to load, rebuild it
        logging.info(f"Building a new vector store for {repo.full_name}")
        documents = fetch_documents_from_repo(repo, github_token)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)

        # Save the latest commit SHA to track changes
        latest_commit = repo.get_commits()[0]
        repo_state = {"latest_commit_sha": latest_commit.sha}
        with open(REPO_STATE_PATH, "w") as f:
            json.dump(repo_state, f)

        logging.info(f"Built and saved vector store for {repo.full_name}")
        return vectorstore

    except Exception as e:
        logging.error(f"Error in load_or_build_vectorstore: {e}")
        # Rebuild the vector store as a last resort and update commit SHA
        documents = fetch_documents_from_repo(repo, github_token)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        # Save the latest commit SHA to track changes
        latest_commit = repo.get_commits()[0]
        repo_state = {"latest_commit_sha": latest_commit.sha}
        with open(REPO_STATE_PATH, "w") as f:
            json.dump(repo_state, f)
        return vectorstore


def fetch_file_content(file_path):
    """
    Fetch the content of a file, handling encoding errors.
    """
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and not mime_type.startswith("text"):
            logging.info(f"Skipping binary file {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
        return {
            "path": file_path,
            "content": content,
        }
    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {e}")
    return None


def fetch_documents_from_repo(repo, github_token):
    """
    Fetch documents from a GitHub repository, including source code.
    """
    documents = []
    repo_dir = f"/tmp/{repo.full_name.replace('/', '_')}"

    try:
        if os.path.exists(repo_dir):
            Repo(repo_dir).remote().pull()
        else:
            Repo.clone_from(repo.clone_url, repo_dir)

        source_files = []
        for root, _, files in os.walk(repo_dir):
            if ".git" in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                source_files.append(file_path)

        source_code_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_file_content, source_files))

        for result in results:
            if result:
                documents.append(
                    Document(
                        page_content=result["content"],
                        metadata={"source": result["path"]},
                    )
                )
                source_code_count += 1

        logging.info(
            f"Fetched {source_code_count} source code files from {repo.full_name}"
        )
    except Exception as e:
        logging.error(f"Error fetching source code from {repo.full_name}: {e}")
    finally:
        # Clean up the temporary directory
        shutil.rmtree(repo_dir)

    return documents


def get_processed_issues(repo):
    """
    Get the set of processed issue numbers from a file specific to the repository.
    """
    PROCESSED_ISSUES_FILE = f"{repo.full_name.replace('/', '_')}_processed_issues.txt"
    with processed_issues_lock:
        if not os.path.exists(PROCESSED_ISSUES_FILE):
            with open(PROCESSED_ISSUES_FILE, "w") as f:
                pass
        with open(PROCESSED_ISSUES_FILE, "r") as f:
            # Acquire shared lock
            fcntl.flock(f, fcntl.LOCK_SH)
            processed = f.read().splitlines()
            # Release lock
            fcntl.flock(f, fcntl.LOCK_UN)
    return set(int(i) for i in processed if i.strip())


def mark_issue_as_processed(repo, issue_number):
    """
    Mark an issue as processed by adding it to the repository-specific file.
    """
    PROCESSED_ISSUES_FILE = f"{repo.full_name.replace('/', '_')}_processed_issues.txt"
    with processed_issues_lock:
        with open(PROCESSED_ISSUES_FILE, "a") as f:
            # Acquire exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(f"{issue_number}\n")
            # Release lock
            fcntl.flock(f, fcntl.LOCK_UN)


def check_and_reply_new_issues(repo, retriever, qa_chain, llm_model_name, start_issue_number):
    """
    Check for new issues in a repository and reply to them using the QA chain.
    """
    processed_issues = get_processed_issues(repo)

    # Get open issues
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        issue_number = issue.number

        # Skip issues with numbers less than start_issue_number
        if issue_number < start_issue_number:
            continue

        if issue_number not in processed_issues:
            # Include both the title and body in the question
            question = f"{issue.title}\n\n{issue.body or ''}"

            if not question.strip():
                # Skip if the issue body is empty
                mark_issue_as_processed(repo, issue_number)
                continue

            try:
                logging.info(f"Processing Issue #{issue_number} in {repo.full_name}")
                # Generate answer using qa_chain
                response = qa_chain.invoke({
                    "input": question
                })

                # Get the answer from the response
                answer = response.get('answer', response.get('result', ''))

                # Append signature to the answer
                signature = (
                    "\n\n---\n*Response generated by 🤖 feifei-bot | ChatGPT-4"
                )
                full_answer = answer.strip() + signature

                logging.info(f"Answer for Issue #{issue_number}: {full_answer}")
                # Post a comment to the issue
                # issue.create_comment(full_answer)

                # Mark issue as processed
                mark_issue_as_processed(repo, issue_number)
                logging.info(f"Replied to Issue #{issue_number} in {repo.full_name}")
            except Exception as e:
                logging.error(
                    f"Error processing Issue #{issue_number} in {repo.full_name}: {e}"
                )


def main():
    github_token = CONFIG.get("github_token")
    if not github_token:
        raise ValueError(
            "GitHub token not found. Please set the GITHUB_TOKEN environment variable."
        )
    g = Github(
        base_url=CONFIG.get("github_base_url", "https://api.github.com"),
        login_or_token=github_token,
    )

    # Get the repositories configuration
    repositories_config = CONFIG.get("repositories", {})
    if not repositories_config:
        raise ValueError(
            "No repositories specified. Please set 'repositories' in the config.yaml file."
        )

    # Initialize embeddings
    embeddings = initialize_embeddings(
        backend=CONFIG.get("embedding_backend"),
        model_name=CONFIG.get("embedding_model_name"),
        embedding_model_name=CONFIG.get("embedding_model_name"),
        openai_api_base=CONFIG.get("openai_api_base"),
        huggingface_api_base=CONFIG.get("huggingface_api_base"),
    )

    # Initialize LLM
    llm = initialize_llm(
        backend=CONFIG.get("llm_backend"),
        model_name=CONFIG.get("model_name"),
        repo_id=CONFIG.get("repo_id"),
        device=CONFIG.get("device"),
        openai_api_base=CONFIG.get("openai_api_base"),
        huggingface_api_base=CONFIG.get("huggingface_api_base"),
    )

    # Get the name of the LLM model for the signature
    llm_model_name = CONFIG.get("model_name")

    # Get the prompt template
    PROMPT = get_prompt_template()
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)

    try:
        while True:
            try:
                logging.info("Checking for new issues...")
                for repo_full_name, repo_settings in repositories_config.items():
                    start_issue_number = repo_settings.get('start_issue_number', 1)
                    repo = g.get_repo(repo_full_name.strip())
                    logging.info(f"Processing repository: {repo_full_name}")

                    # Load or build vector store
                    vectorstore = load_or_build_vectorstore(
                        repo, embeddings, github_token
                    )
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                    # Create the RetrievalQA chain with the initialized retriever
                    qa_chain = create_retrieval_chain(
                        combine_docs_chain=combine_docs_chain,
                        retriever=retriever,
                    )

                    # Check and reply to new issues
                    check_and_reply_new_issues(
                        repo, retriever, qa_chain, llm_model_name, start_issue_number
                    )

                logging.info(
                    f"Done. Sleeping for {CONFIG.get('check_interval', 300)} seconds."
                )
            except Exception as e:
                logging.error(f"An error occurred: {e}")

            # Wait for the specified interval before checking again
            time.sleep(CONFIG.get("check_interval", 300))
    except KeyboardInterrupt:
        logging.info("Program terminated by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

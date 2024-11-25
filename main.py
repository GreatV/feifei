import os
import sys
import time
import json
import logging
import concurrent.futures
from github import Github
from git import Repo
import shutil
import portalocker
from dotenv import load_dotenv
import yaml
import argparse
from ratelimit import limits, sleep_and_retry
import textwrap

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from a .env file if present
load_dotenv()


# Configure logging
logging_level = "INFO"
logging.basicConfig(
    level=getattr(logging, logging_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("feifei-bot.log"), logging.StreamHandler()],
)

# Define rate limits (e.g., 5000 requests per hour for GitHub)
GITHUB_RATE_LIMIT = 5000
GITHUB_RATE_PERIOD = 3600  # 1 hour

# Define rate limits for OpenAI API if applicable
OPENAI_RATE_LIMIT = 60
OPENAI_RATE_PERIOD = 60  # 1 minute


# Decorator to handle GitHub API rate limiting
@sleep_and_retry
@limits(calls=GITHUB_RATE_LIMIT, period=GITHUB_RATE_PERIOD)
def get_issues_with_rate_limit(repo, state="open"):
    return repo.get_issues(state=state)


# Decorator to handle OpenAI API rate limiting
@sleep_and_retry
@limits(calls=OPENAI_RATE_LIMIT, period=OPENAI_RATE_PERIOD)
def openai_api_request(qa_chain, question):
    return qa_chain.invoke({"input": question})


# Load configuration from YAML file
def load_config(yaml_file="config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {yaml_file}")
    except FileNotFoundError:
        logging.error(f"Configuration file {yaml_file} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    return config


def validate_config(config):
    """
    Validate the configuration dictionary to ensure all required keys are present.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    required_keys = [
        "github_base_url",
        "repositories",
        "embedding_backend",
        "llm_backend",
        "model_name",
        "signature",
        "prompt_template",
    ]
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            sys.exit(1)
    logging.info("Configuration validation passed.")


def initialize_llm(backend="openai", **kwargs):
    """
    Initialize the Language Model (LLM) based on the selected backend.

    Args:
        backend (str): The backend to use for the LLM.
        **kwargs: Additional keyword arguments for the LLM initialization.

    Returns:
        object: Initialized LLM object.
    """
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=kwargs.get("model_name", "gpt-3.5-turbo"),
            openai_api_base=kwargs.get("openai_api_base"),
        )
    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-v1"),
            max_tokens_to_sample=kwargs.get("max_tokens_to_sample", 1024),
        )
    elif backend == "google":
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(
            model_name=kwargs.get("model_name", "chat-bison"),
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_output_tokens", 1024),
        )
    elif backend == "qianfan":
        from langchain_community.chat_models import QianfanChatEndpoint

        llm = QianfanChatEndpoint(
            model=kwargs.get("model_name", "ERNIE-3.5-8K"),
            temperature=kwargs.get("temperature", 0.7),
        )
    elif backend == "ollama":
        from langchain_ollama.chat_models import ChatOllama

        llm = ChatOllama(
            base_url=kwargs.get("ollama_base_url", "http://localhost:11434"),
            model=kwargs.get("model_name", "llama2"),
        )
    elif backend == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=kwargs.get("repo_id", "gpt2"),
            model_kwargs={
                "temperature": kwargs.get("temperature", 0.7),
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
            },
        )
    elif backend == "local":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_name = kwargs.get("model_name", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            device=kwargs.get("device", -1),
        )

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")
    return llm


def initialize_embeddings(backend="openai", **kwargs):
    """
    Initialize the embeddings model based on the selected backend.

    Args:
        backend (str): The backend to use for the embeddings.
        **kwargs: Additional keyword arguments for the embeddings initialization.

    Returns:
        object: Initialized embeddings object.
    """
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=kwargs.get("model_name"),
            openai_api_base=kwargs.get("openai_api_base"),
        )
    elif backend == "qianfan":
        from langchain_community.embeddings import QianfanEmbeddingsEndpoint

        embeddings = QianfanEmbeddingsEndpoint()
    elif backend == "huggingface" or backend == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = kwargs.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    return embeddings


def get_prompt_template(prompt_template: str):
    """
    Retrieve the prompt template.

    Args:
        prompt_template (str): The prompt template string.

    Returns:
        PromptTemplate: The prompt template object.
    """
    return PromptTemplate(
        template=textwrap.dedent(prompt_template), input_variables=["input", "context"]
    )


def rebuild_vectorstore(
    repo, embeddings, github_token, VECTORSTORE_PATH, REPO_STATE_PATH, branch="main"
):
    """
    Rebuild the vector store and update the repository state.

    Args:
        repo (Repository): The GitHub repository object.
        embeddings (object): The embeddings object.
        github_token (str): GitHub token for authentication.
        VECTORSTORE_PATH (str): Path to save the vector store.
        REPO_STATE_PATH (str): Path to save the repository state.
        branch (str): Branch to fetch documents from.

    Returns:
        FAISS: The rebuilt vector store object.
    """
    documents = fetch_documents_from_repo(repo, github_token, branch)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    # Save the latest commit SHA to track changes
    latest_commit = repo.get_commits()[0]
    repo_state = {"latest_commit_sha": latest_commit.sha}
    with open(REPO_STATE_PATH, "w") as f:
        json.dump(repo_state, f)

    logging.info(f"Built and saved vector store for {repo.full_name}")
    return vectorstore


def load_or_build_vectorstore(repo, embeddings, github_token, config, branch="main"):
    """
    Load an existing vector store or build a new one if the repository has new commits.

    Args:
        repo (Repository): The GitHub repository object.
        embeddings (object): The embeddings object.
        github_token (str): GitHub token for authentication.
        config (dict): Configuration dictionary.
        branch (str): Branch to fetch documents from.

    Returns:
        FAISS: The loaded or rebuilt vector store object.
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
        return rebuild_vectorstore(
            repo, embeddings, github_token, VECTORSTORE_PATH, REPO_STATE_PATH, branch
        )

    except Exception as e:
        logging.error(f"Error in load_or_build_vectorstore: {e}")
        # Rebuild the vector store as a last resort and update commit SHA
        return rebuild_vectorstore(
            repo, embeddings, github_token, VECTORSTORE_PATH, REPO_STATE_PATH, branch
        )


def is_binary_file(filepath):
    """
    Check if a file is binary.

    Args:
        filepath (str): Path to the file.

    Returns:
        bool: True if the file is binary, False otherwise.
    """
    with open(filepath, "rb") as file:
        initial_bytes = file.read(1024)
        if b"\0" in initial_bytes:
            return True
    return False


def fetch_file_content(file_path, repo_url, branch):
    """
    Fetch the content of a file.

    Args:
        file_path (str): Path to the file.
        repo_url (str): URL of the repository.
        branch (str): Branch to fetch the file from.

    Returns:
        dict: Dictionary containing file path, content, and URL.
    """
    if is_binary_file(file_path):
        logging.info(f"Skipping binary file {file_path}")
        return None

    try:
        with open(file_path, "r") as file:
            content = file.read()
        return {
            "path": file_path,
            "content": content,
            "url": f"{repo_url}/blob/{branch}/{os.path.relpath(file_path, start=os.path.dirname(repo_url))}",
        }
    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {e}")
    return None


def fetch_documents_from_repo(repo, github_token, branch=None):
    """
    Fetch documents from a GitHub repository, including source code.

    Args:
        repo (Repository): The GitHub repository object.
        github_token (str): GitHub token for authentication.
        branch (str): Branch to fetch documents from.

    Returns:
        list: List of Document objects.
    """
    documents = []
    repo_dir = f"/tmp/{repo.full_name.replace('/', '_')}"
    repo_url = repo.html_url

    # Use the branch specified in config.yaml or the default branch from GitHub API
    if branch is None:
        branch = repo.default_branch

    try:
        if os.path.exists(repo_dir):
            Repo(repo_dir).remote().pull()
            Repo(repo_dir).git.checkout(branch)
        else:
            Repo.clone_from(repo.clone_url, repo_dir, branch=branch)

        source_files = []
        for root, _, files in os.walk(repo_dir):
            if ".git" in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                source_files.append(file_path)

        source_code_count = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda file_path: fetch_file_content(file_path, repo_url, branch),
                    source_files,
                )
            )

        for result in results:
            if result:
                documents.append(
                    Document(
                        page_content=result["content"],
                        metadata={"source": result["path"], "url": result["url"]},
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

    Args:
        repo (Repository): The GitHub repository object.

    Returns:
        set: Set of processed issue numbers.
    """
    PROCESSED_ISSUES_FILE = f"{repo.full_name.replace('/', '_')}_processed_issues.txt"
    if not os.path.exists(PROCESSED_ISSUES_FILE):
        with open(PROCESSED_ISSUES_FILE, "w") as f:
            pass
    with open(PROCESSED_ISSUES_FILE, "r") as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        processed = f.read().splitlines()
        portalocker.unlock(f)
    return set(int(i) for i in processed if i.strip())


def mark_issue_as_processed(repo, issue_number):
    """
    Mark an issue as processed by adding it to the repository-specific file.

    Args:
        repo (Repository): The GitHub repository object.
        issue_number (int): The issue number to mark as processed.

    Returns:
        None
    """
    PROCESSED_ISSUES_FILE = f"{repo.full_name.replace('/', '_')}_processed_issues.txt"
    with open(PROCESSED_ISSUES_FILE, "a") as f:
        # Acquire exclusive lock
        portalocker.lock(f, portalocker.LOCK_EX)
        f.write(f"{issue_number}\n")
        # Release lock
        portalocker.unlock(f)


def check_and_reply_new_issues(
    repo,
    retriever,
    qa_chain,
    llm_model_name,
    start_issue_number,
    debug,
    signature_template,
    max_retries=3,
    retry_interval=60,
):
    """
    Check for new issues in a repository and reply to them using the QA chain.

    Args:
        repo (Repository): The GitHub repository object.
        retriever (object): The retriever object.
        qa_chain (object): The QA chain object.
        llm_model_name (str): The name of the LLM model.
        start_issue_number (int): The starting issue number to check.
        debug (bool): Debug mode flag.
        signature_template (str): The signature template string.
        max_retries (int): Maximum number of retries for processing an issue.
        retry_interval (int): Interval between retries in seconds.

    Returns:
        None
    """
    processed_issues = get_processed_issues(repo)

    # Get open issues
    open_issues = get_issues_with_rate_limit(repo, state="open")

    for issue in open_issues:
        issue_number = issue.number

        # Skip issues with numbers less than start_issue_number
        if issue_number < start_issue_number:
            continue

        # Skip pull requests
        if issue.pull_request is not None:
            logging.info(f"Skipping PR #{issue_number} in {repo.full_name}")
            mark_issue_as_processed(repo, issue_number)
            continue

        if issue_number not in processed_issues:
            # Include both the title and body in the question
            question = f"{issue.title}\n\n{issue.body or ''}"

            if not question.strip():
                # Log and skip if both the issue title and body are empty
                logging.info(
                    f"Skipping Issue #{issue_number} in {repo.full_name} because it has no content."
                )
                mark_issue_as_processed(repo, issue_number)
                continue

            retries = 0
            while retries < max_retries:
                try:
                    logging.info(
                        f"Processing Issue #{issue_number} in {repo.full_name}"
                    )
                    # Generate answer using qa_chain
                    response = openai_api_request(qa_chain, question)

                    # Get the answer from the response
                    answer = response.get("answer", response.get("result", ""))

                    # Append signature to the answer
                    signature = signature_template.format(model_name=llm_model_name)
                    full_answer = answer.strip() + signature

                    logging.info(f"Answer for Issue #{issue_number}: {full_answer}")

                    if not debug:
                        # Post a comment to the issue
                        issue.create_comment(full_answer)
                        logging.info(
                            f"Replied to Issue #{issue_number} in {repo.full_name}"
                        )
                    else:
                        logging.info(
                            f"Debug mode: Would have commented on Issue #{issue_number}"
                        )

                    # Mark issue as processed
                    mark_issue_as_processed(repo, issue_number)
                    break
                except Exception as e:
                    retries += 1
                    logging.error(
                        f"Error processing Issue #{issue_number} in {repo.full_name}: {e}. Retry {retries}/{max_retries}"
                    )
                    if retries == max_retries:
                        logging.error(
                            f"Failed to process Issue #{issue_number} in {repo.full_name} after {max_retries} retries."
                        )
                    else:
                        time.sleep(retry_interval)  # Wait before retrying


def process_repository(
    repo_full_name,
    repo_settings,
    embeddings,
    combine_docs_chain,
    github_token,
    config,
    args,
    llm_model_name,
    github_client,
    signature_template,
):
    """
    Process a repository by loading or building the vector store and checking for new issues.

    Args:
        repo_full_name (str): Full name of the repository.
        repo_settings (dict): Repository-specific settings.
        embeddings (object): The embeddings object.
        combine_docs_chain (object): The combined documents chain object.
        github_token (str): GitHub token for authentication.
        config (dict): Configuration dictionary.
        args (Namespace): Command-line arguments.
        llm_model_name (str): The name of the LLM model.
        github_client (Github): The GitHub client object.
        signature_template (str): The signature template string.

    Returns:
        None
    """
    try:
        start_issue_number = repo_settings.get("start_issue_number", 1)
        branch = repo_settings.get("branch", None)
        repo = github_client.get_repo(repo_full_name.strip())
        logging.info(f"Processing repository: {repo_full_name}")

        # Load or build vector store
        vectorstore = load_or_build_vectorstore(
            repo, embeddings, github_token, config, branch
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Create retrieval chain
        qa_chain = create_retrieval_chain(
            combine_docs_chain=combine_docs_chain,
            retriever=retriever,
        )

        # Check and reply to new issues
        check_and_reply_new_issues(
            repo,
            retriever,
            qa_chain,
            llm_model_name,
            start_issue_number,
            args.debug,
            signature_template,
            max_retries=3,
        )
    except Exception as e:
        logging.error(
            f"An error occurred while processing repository {repo_full_name}: {e}"
        )


def main():
    """
    Main function to process GitHub issues.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process GitHub issues.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only output logs, do not actually comment on issues.",
    )
    args = parser.parse_args()

    # Load configurations
    config = load_config()

    # Validate configurations
    validate_config(config)

    github_token = os.getenv("GITHUB_TOKEN", None)
    if not github_token:
        raise ValueError(
            "GitHub token not found. Please set the GITHUB_TOKEN environment variable."
        )
    github_client = Github(
        base_url=config.get("github_base_url", "https://api.github.com"),
        login_or_token=github_token,
    )

    # Get the repositories configuration
    repositories_config = config.get("repositories", {})
    if not repositories_config:
        raise ValueError(
            "No repositories specified. Please set 'repositories' in the config.yaml file."
        )

    # Initialize embeddings
    embeddings = initialize_embeddings(
        backend=config.get("embedding_backend"),
        model_name=config.get("embedding_model_name"),
        embedding_model_name=config.get("embedding_model_name"),
        openai_api_base=config.get("openai_api_base"),
        huggingface_api_base=config.get("huggingface_api_base"),
    )

    # Initialize LLM
    llm = initialize_llm(
        backend=config.get("llm_backend"),
        model_name=config.get("model_name"),
        repo_id=config.get("repo_id"),
        device=config.get("device"),
        openai_api_base=config.get("openai_api_base"),
        huggingface_api_base=config.get("huggingface_api_base"),
    )

    # Get the name of the LLM model for the signature
    llm_model_name = config.get("model_name")

    # Load signature template
    signature_template = config.get("signature")

    # Get the prompt template
    PROMPT = get_prompt_template(config.get("prompt_template"))
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)

    try:
        while True:
            try:
                logging.info("Checking for new issues...")

                # Process repositories concurrently
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_repo = {
                        executor.submit(
                            process_repository,
                            repo_full_name,
                            repo_settings,
                            embeddings,
                            combine_docs_chain,
                            github_token,
                            config,
                            args,
                            llm_model_name,
                            github_client,
                            signature_template,
                        ): repo_full_name
                        for repo_full_name, repo_settings in repositories_config.items()
                    }
                    # Wait for all futures to complete
                    for future in concurrent.futures.as_completed(future_to_repo):
                        repo_full_name = future_to_repo[future]
                        try:
                            future.result()
                        except Exception as exc:
                            logging.error(
                                f"{repo_full_name} generated an exception: {exc}"
                            )

                logging.info(
                    f"Done. Sleeping for {config.get('check_interval', 600)} seconds."
                )
            except Exception as e:
                logging.error(f"An error occurred: {e}")
            finally:
                # Wait for the specified interval before checking again
                time.sleep(config.get("check_interval", 600))
    except KeyboardInterrupt:
        logging.info("Program terminated by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

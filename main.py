import os
import sys
import time
import json
import logging
import concurrent.futures
import datetime
from github import Github
from git import Repo
import portalocker
from dotenv import load_dotenv
import yaml
import argparse
import textwrap
import requests
import pickle

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


def load_config(yaml_file="config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Configuration loaded from {yaml_file}")
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
    logging.info(f"Initialized LLM with backend: {backend}")
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
    logging.info(f"Initialized embeddings with backend: {backend}")
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
        template=textwrap.dedent(prompt_template),
        input_variables=["input", "context", "sources"],
    )


def rebuild_vectorstore(
    repo,
    embeddings,
    github_token,
    vectorstore_path,
    repo_state_path,
    branch="main",
    recent_period=None,
):
    """
    Rebuild the vector store and update the repository state.

    Args:
        repo (Repository): The GitHub repository object.
        embeddings (object): The embeddings object.
        github_token (str): GitHub token for authentication.
        vectorstore_path (str): Path to save the vector store.
        repo_state_path (str): Path to save the repository state.
        branch (str): Branch to fetch documents from.
        recent_period (dict): Dictionary with keys 'months' or 'weeks' to filter documents.

    Returns:
        FAISS: The rebuilt vector store object.
    """
    documents = fetch_documents_from_repo(repo, github_token, branch, recent_period)
    batch_size = 10  # Define batch size
    docmument_size = len(documents)
    init_size = batch_size if docmument_size > batch_size else docmument_size
    vectorstore = FAISS.from_documents(documents[:init_size], embeddings)
    for i in range(init_size, len(documents), batch_size):
        end_size = i + batch_size if i + batch_size < docmument_size else docmument_size
        batch_documents = documents[i:end_size]
        vectorstore.add_documents(batch_documents)
    vectorstore.save_local(vectorstore_path)

    # Save the latest commit SHA to track changes
    latest_commit = repo.get_commits()[0]
    repo_state = {"latest_commit_sha": latest_commit.sha}
    with open(repo_state_path, "w") as f:
        json.dump(repo_state, f)

    logging.info(f"Built and saved vector store for {repo.full_name}")
    return vectorstore


def load_or_build_vectorstore(
    repo, embeddings, github_token, config, branch="main", recent_period=None
):
    embeddings_model_name = config.get("embedding_model_name", "default")
    vectorstore_file = f"{repo.full_name.replace('/', '_')}_{embeddings_model_name.replace('/', '_')}_vectorstore"
    repo_state_file = f"{repo.full_name.replace('/', '_')}_repo_state.json"
    vectorstore_path = os.path.join(".cache", vectorstore_file)
    repo_state_path = os.path.join(".cache", repo_state_file)

    if os.path.exists(vectorstore_path) and os.path.getsize(vectorstore_path) > 0:
        with open(repo_state_path, "r") as f:
            repo_state = json.load(f)

        latest_commit_sha = repo.get_commits()[0].sha

        if repo_state.get("latest_commit_sha") == latest_commit_sha:
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logging.info(f"Loaded existing vector store for {repo.full_name}")
            return vectorstore

    logging.info(
        f"No existing vector store found or repository has new commits. Building a new one for {repo.full_name}"
    )
    return rebuild_vectorstore(
        repo,
        embeddings,
        github_token,
        vectorstore_path,
        repo_state_path,
        branch,
        recent_period,
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


def fetch_file_content(file_path, repo_url, repo_dir, branch):
    """
    Fetch the content of a file.

    Args:
        file_path (str): Path to the file.
        repo_url (str): URL of the repository.
        repo_dir (str): Local directory of the repository.
        branch (str): Branch name.

    Returns:
        dict: Dictionary containing file path, content, and URL.
    """
    if is_binary_file(file_path):
        logging.info(f"Skipping binary file {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()
    relative_path = os.path.relpath(file_path, start=repo_dir)
    return {
        "path": file_path,
        "content": content,
        "url": f"{repo_url}/blob/{branch}/{relative_path}",
    }


def fetch_source_code_documents(repo, github_token, branch):
    """
    Fetch source code documents from the repository.

    Args:
        repo (Repository): The GitHub repository object.
        github_token (str): GitHub token.
        branch (str): Branch name.

    Returns:
        list: List of Document objects containing source code.
    """
    documents = []
    repo_dir = os.path.join(".cache", repo.full_name.replace("/", "_"))
    repo_url = repo.html_url

    # Use the branch specified in config.yaml or the default branch from GitHub API
    if branch is None:
        branch = repo.default_branch

    if os.path.exists(repo_dir):
        repo_local = Repo(repo_dir)
        repo_local.git.checkout(branch)
        repo_local.remotes.origin.pull()
        logging.info(f"Pulled latest code for {repo.full_name}")
    else:
        Repo.clone_from(repo.clone_url, repo_dir, branch=branch)
        logging.info(f"Cloned repository {repo.full_name}")

    source_files = []
    for root, _, files in os.walk(repo_dir):
        if ".git" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            source_files.append(file_path)

    source_code_count = 0
    for file_path in source_files:
        result = fetch_file_content(file_path, repo_url, repo_dir, branch)
        if result:
            documents.append(
                Document(
                    page_content=result["content"],
                    metadata={"source": result["path"], "url": result["url"]},
                )
            )
            source_code_count += 1

    logging.info(f"Fetched {source_code_count} source code files from {repo.full_name}")
    return documents


def cache_data(data, cache_file):
    """
    Cache data to a local file.

    Args:
        data (list): List of Document objects to cache.
        cache_file (str): Path to the cache file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"Cached data to {cache_file}")


def load_cached_data(cache_file):
    """
    Load cached data from a local file.

    Args:
        cache_file (str): Path to the cache file.

    Returns:
        list: List of Document objects.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Loaded cached data from {cache_file}")
        return data
    return []


def calculate_since_date(recent_period):
    if recent_period:
        if "months" in recent_period:
            return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                days=30 * recent_period["months"]
            )
        elif "weeks" in recent_period:
            return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                weeks=recent_period["weeks"]
            )
    return None


def fetch_existing_issues(repo, recent_period=None):
    """
    Fetch existing issues from the repository, filtered by creation date.

    Args:
        repo (Repository): GitHub repository object.
        recent_period (dict): Dictionary with keys 'months' or 'weeks' to filter issues.

    Returns:
        list: List of Document objects created from issues.
    """
    cache_file = f"{repo.full_name.replace('/', '_')}_issues_cache.pkl"
    cache_file = os.path.join(".cache", cache_file)
    documents = load_cached_data(cache_file)
    if documents:
        return documents

    documents = []
    since = calculate_since_date(recent_period)

    # Fetch issues (consider pagination)
    issues = repo.get_issues(state="all", sort="created", direction="desc")

    for issue in issues:
        # Check if issue was created after 'since' date
        if since and issue.created_at < since:
            break  # Since issues are sorted by creation date descending, we can stop here

        logging.info(f"Fetching Issue #{issue.number} from {repo.full_name}")
        # Skip pull requests
        if issue.pull_request is not None:
            continue
        content = f"Issue Title: {issue.title}\n\n{issue.body or ''}"
        metadata = {
            "source": f"Issue #{issue.number}",
            "url": issue.html_url,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "comments": issue.comments,
        }
        documents.append(Document(page_content=content, metadata=metadata))

    logging.info(f"Fetched {len(documents)} issues from {repo.full_name}")
    cache_data(documents, cache_file)
    return documents


def fetch_existing_discussions(repo, github_token, recent_period=None):
    """
    Fetch existing discussions from the repository, filtered by creation date.

    Args:
        repo (Repository): GitHub repository object.
        github_token (str): GitHub token.
        recent_period (dict): Dictionary with keys 'months' or 'weeks' to filter discussions.

    Returns:
        list: List of Document objects created from discussions.
    """
    cache_file = f"{repo.full_name.replace('/', '_')}_discussions_cache.pkl"
    cache_file = os.path.join(".cache", cache_file)
    documents = load_cached_data(cache_file)
    if documents:
        return documents

    documents = []
    since = calculate_since_date(recent_period)

    # Use GitHub GraphQL API to fetch discussions with pagination
    has_next_page = True
    end_cursor = None

    while has_next_page:
        query = """
        query($owner: String!, $name: String!, $after: String) {
          repository(owner: $owner, name: $name) {
            discussions(first: 100, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
              pageInfo {
                endCursor
                hasNextPage
              }
              nodes {
                number
                title
                body
                url
                createdAt
                updatedAt
              }
            }
          }
        }
        """

        variables = {
            "owner": repo.owner.login,
            "name": repo.name,
            "after": end_cursor,
        }

        headers = {
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={"query": query, "variables": variables},
        )

        if response.status_code == 200:
            data = response.json()
            repository = data.get("data", {}).get("repository", {})
            if not repository:
                logging.error(f"No repository data found for {repo.full_name}")
                break

            discussions = repository.get("discussions", {}).get("nodes", [])
            page_info = repository.get("discussions", {}).get("pageInfo", {})
            has_next_page = page_info.get("hasNextPage", False)
            end_cursor = page_info.get("endCursor", None)

            for discussion in discussions:
                # Filter discussions based on recent_period
                discussion_created_at = datetime.datetime.fromisoformat(
                    discussion["createdAt"].rstrip("Z")
                ).replace(tzinfo=datetime.timezone.utc)
                if since and discussion_created_at < since:
                    has_next_page = False  # No need to fetch older discussions
                    break

                logging.info(
                    f"Fetching Discussion #{discussion['number']} from {repo.full_name}"
                )
                content = (
                    f"Discussion Title: {discussion['title']}\n\n{discussion['body']}"
                )
                metadata = {
                    "source": f"Discussion #{discussion['number']}",
                    "url": discussion["url"],
                    "created_at": discussion["createdAt"],
                    "updated_at": discussion["updatedAt"],
                }
                documents.append(Document(page_content=content, metadata=metadata))
            logging.info(f"Fetched {len(documents)} discussions from {repo.full_name}")
        else:
            logging.error(
                f"Failed to fetch discussions from {repo.full_name}: {response.status_code}, {response.text}"
            )
            break  # Exit the loop on error
    cache_data(documents, cache_file)
    return documents


def fetch_documents_from_repo(repo, github_token, branch=None, recent_period=None):
    """
    Fetch documents from a GitHub repository, including source code, issues, and discussions.

    Args:
        repo (Repository): The GitHub repository object.
        github_token (str): GitHub token for authentication.
        branch (str): Branch to fetch documents from.
        recent_period (dict): Dictionary with keys 'months' or 'weeks' to filter issues and discussions.

    Returns:
        list: List of Document objects.
    """
    documents = []

    # Fetch source code files
    source_documents = fetch_source_code_documents(repo, github_token, branch)
    documents.extend(source_documents)

    # Fetch existing issues
    issue_documents = fetch_existing_issues(repo, recent_period)
    documents.extend(issue_documents)

    # Fetch existing discussions
    discussion_documents = fetch_existing_discussions(repo, github_token, recent_period)
    documents.extend(discussion_documents)

    logging.info(f"Fetched total of {len(documents)} documents from {repo.full_name}")
    return documents


def get_processed_issues(repo):
    """
    Get the set of processed issue numbers from a file specific to the repository.

    Args:
        repo (Repository): The GitHub repository object.

    Returns:
        set: Set of processed issue numbers.
    """
    processed_issues_file = os.path.join(
        ".cache", f"{repo.full_name.replace('/', '_')}_processed_issues.txt"
    )
    os.makedirs(os.path.dirname(processed_issues_file), exist_ok=True)
    if not os.path.exists(processed_issues_file):
        with open(processed_issues_file, "w") as f:
            pass
    with open(processed_issues_file, "r") as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        processed = f.read().splitlines()
        portalocker.unlock(f)
    logging.info(f"Processed issues retrieved for {repo.full_name}")
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
    processed_issues_file = os.path.join(
        ".cache", f"{repo.full_name.replace('/', '_')}_processed_issues.txt"
    )
    os.makedirs(os.path.dirname(processed_issues_file), exist_ok=True)
    with open(processed_issues_file, "a") as f:
        # Acquire exclusive lock
        portalocker.lock(f, portalocker.LOCK_EX)
        f.write(f"{issue_number}\n")
        # Release lock
        portalocker.unlock(f)
    logging.info(f"Issue #{issue_number} marked as processed for {repo.full_name}")


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
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        issue_number = issue.number

        # Skip issues with numbers less than start_issue_number
        if issue_number <= start_issue_number:
            continue

        # Skip issues that have already been processed
        if issue_number in processed_issues:
            continue

        # Skip pull requests
        if issue.pull_request is not None:
            logging.info(f"Skipping PR #{issue_number} in {repo.full_name}")
            mark_issue_as_processed(repo, issue_number)
            continue

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
                logging.info(f"Processing Issue #{issue_number} in {repo.full_name}")
                # Generate answer using qa_chain
                response = qa_chain.invoke({"input": question})
                answer = response.get("answer", response.get("result", ""))

                # Append signature and sources to the answer
                signature = signature_template.format(model_name=llm_model_name)
                full_answer = f"{answer.strip()}{signature}"

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

    logging.info(f"Checked and replied to new issues for {repo.full_name}")


def handle_cuda_oom_error():
    """
    Handle CUDA out of memory error by setting the PYTORCH_CUDA_ALLOC_CONF environment variable.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logging.info(
        "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to handle CUDA OOM error."
    )


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
    start_issue_number = repo_settings.get("start_issue_number", 1)
    branch = repo_settings.get("branch", None)
    recent_period = repo_settings.get("recent_period", None)
    repo = github_client.get_repo(repo_full_name.strip())
    logging.info(f"Processing repository: {repo_full_name}")

    # Load or build vector store
    vectorstore = load_or_build_vectorstore(
        repo, embeddings, github_token, config, branch, recent_period
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

    while True:
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
                future_to_repo[future]
                future.result()

        logging.info(f"Done. Sleeping for {config.get('check_interval', 600)} seconds.")
        # Wait for the specified interval before checking again
        time.sleep(config.get("check_interval", 600))


if __name__ == "__main__":
    main()

import os
import datetime
import logging
import requests
import pickle
import portalocker
from git import Repo
from langchain_core.documents import Document
import time
from functools import wraps

def retry_on_connection_error(max_retries=3, delay=1):
    """
    Decorator to retry functions on connection errors.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, 
                       requests.exceptions.Timeout,
                       requests.exceptions.RequestException) as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logging.warning(f"Connection error: {str(e)}. Retrying... ({retries}/{max_retries})")
                    time.sleep(delay * retries)  # Exponential backoff
            return None
        return wrapper
    return decorator

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

@retry_on_connection_error()
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
        content = f"Issue Title: {issue.title}\nIssue URL: {issue.html_url}\n\n{issue.body or ''}"
        metadata = {
            "source": f"Issue #{issue.number}",
            "title": issue.title,
            "url": issue.html_url,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "comments": issue.comments,
        }
        documents.append(Document(page_content=content, metadata=metadata))

    logging.info(f"Fetched {len(documents)} issues from {repo.full_name}")
    cache_data(documents, cache_file)
    return documents

@retry_on_connection_error()
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
                content = f"Discussion Title: {discussion['title']}\nDiscussion URL: {discussion['url']}\n\n{discussion['body']}"
                metadata = {
                    "source": f"Discussion #{discussion['number']}",
                    "title": discussion["title"],
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
                logging.info(f"Replied to Issue #{issue_number} in {repo.full_name}")
            else:
                logging.info(
                    f"Debug mode: Would have commented on Issue #{issue_number}"
                )

            # Mark issue as processed
            mark_issue_as_processed(repo, issue_number)
        except Exception as e:
            logging.error(
                f"Error processing Issue #{issue_number} in {repo.full_name}: {e}"
            )

    logging.info(f"Checked and replied to new issues for {repo.full_name}")


def get_processed_discussions(repo):
    """
    Get the set of processed discussion numbers from a file specific to the repository.

    Args:
        repo (Repository): The GitHub repository object.

    Returns:
        set: Set of processed discussion numbers.
    """
    processed_discussions_file = os.path.join(
        ".cache", f"{repo.full_name.replace('/', '_')}_processed_discussions.txt"
    )
    os.makedirs(os.path.dirname(processed_discussions_file), exist_ok=True)
    if not os.path.exists(processed_discussions_file):
        with open(processed_discussions_file, "w") as f:
            pass
    with open(processed_discussions_file, "r") as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        processed = f.read().splitlines()
        portalocker.unlock(f)
    logging.info(f"Processed discussions retrieved for {repo.full_name}")
    return set(int(i) for i in processed if i.strip())


def mark_discussion_as_processed(repo, discussion_number):
    """
    Mark a discussion as processed by adding it to the repository-specific file.

    Args:
        repo (Repository): The GitHub repository object.
        discussion_number (int): The discussion number to mark as processed.

    Returns:
        None
    """
    processed_discussions_file = os.path.join(
        ".cache", f"{repo.full_name.replace('/', '_')}_processed_discussions.txt"
    )
    os.makedirs(os.path.dirname(processed_discussions_file), exist_ok=True)
    with open(processed_discussions_file, "a") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        f.write(f"{discussion_number}\n")
        portalocker.unlock(f)
    logging.info(
        f"Discussion #{discussion_number} marked as processed for {repo.full_name}"
    )

@retry_on_connection_error()
def check_and_reply_new_discussions(
    repo,
    retriever,
    qa_chain,
    llm_model_name,
    start_discussion_number,
    debug,
    signature_template,
    github_token,
):
    """
    Check for new discussions in a repository and reply to them using the QA chain.

    Args:
        repo (Repository): The GitHub repository object.
        retriever (object): The retriever object.
        qa_chain (object): The QA chain object.
        llm_model_name (str): The name of the LLM model.
        start_discussion_number (int): The starting discussion number to check.
        debug (bool): Debug mode flag.
        signature_template (str): The signature template string.
        github_token (str): GitHub token for authentication.

    Returns:
        None
    """
    processed_discussions = get_processed_discussions(repo)

    # Use GitHub GraphQL API to fetch discussions
    has_next_page = True
    end_cursor = None

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json",
    }

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
                comments(first: 100) {
                  totalCount
                }
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
                discussion_number = discussion["number"]

                if discussion_number <= start_discussion_number:
                    continue

                if discussion_number in processed_discussions:
                    continue

                question = f"{discussion['title']}\n\n{discussion['body'] or ''}"

                if not question.strip():
                    logging.info(
                        f"Skipping Discussion #{discussion_number} because it has no content."
                    )
                    mark_discussion_as_processed(repo, discussion_number)
                    continue

                # try:
                logging.info(
                    f"Processing Discussion #{discussion_number} in {repo.full_name}"
                )
                response = qa_chain.invoke({"input": question})
                answer = response.get("answer", response.get("result", ""))

                signature = signature_template.format(model_name=llm_model_name)
                full_answer = f"{answer.strip()}{signature}"

                logging.info(
                    f"Answer for Discussion #{discussion_number}: {full_answer}"
                )

                discussion_schema = """
                    id
                    number
                    title
                    body
                    createdAt
                    updatedAt
                    comments(first: 100) {
                        nodes {
                            id
                            body
                        }
                        totalCount
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                    }
                """

                if not debug:
                    # Post a comment to the discussion
                    discussion_obj = repo.get_discussion(
                        number=discussion_number,
                        discussion_graphql_schema=discussion_schema,
                    )
                    discussion_obj.add_comment(
                        body=full_answer,
                        output_schema="""
                            id
                            body
                            createdAt
                        """,
                    )
                    logging.info(
                        f"Replied to Discussion #{discussion_number} in {repo.full_name}"
                    )
                else:
                    logging.info(
                        f"Debug mode: Would have commented on Discussion #{discussion_number}"
                    )

                mark_discussion_as_processed(repo, discussion_number)
            # except Exception as e:
            #     logging.error(
            #         f"Error processing Discussion #{discussion_number} in {repo.full_name}: {e}"
            #     )

        else:
            logging.error(
                f"Failed to fetch discussions from {repo.full_name}: {response.status_code}, {response.text}"
            )
            break

    logging.info(f"Checked and replied to new discussions for {repo.full_name}")

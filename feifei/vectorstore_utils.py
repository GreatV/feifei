import os
import json
import logging
from langchain_community.vectorstores import FAISS
from .github_utils import fetch_documents_from_repo


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
        embeddings (object): The embeddings model object.
        github_token (str): GitHub token for authentication.
        vectorstore_path (str): Path to save the vector store.
        repo_state_path (str): Path to save the repository state.
        branch (str, optional): Branch to fetch documents from. Defaults to "main".
        recent_period (dict, optional): Dictionary with keys 'months' or 'weeks' to filter documents.

    Returns:
        FAISS: The rebuilt vector store object.
    """
    documents = fetch_documents_from_repo(repo, github_token, branch, recent_period)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        if i == 0:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        # force garbage collection every 5 batches
        if i % (batch_size * 5) == 0:
            import gc

            gc.collect()

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
    """
    Load an existing vector store or build a new one if it doesn't exist.

    Args:
        repo (Repository): The GitHub repository object.
        embeddings (object): The embeddings model object.
        github_token (str): GitHub token for authentication.
        config (dict): Configuration dictionary.
        branch (str, optional): Branch to fetch documents from. Defaults to "main".
        recent_period (dict, optional): Dictionary with keys 'months' or 'weeks' to filter documents.

    Returns:
        FAISS: The loaded or newly built vector store object.
    """
    embeddings_model_name = config.get("embedding_model_name", "default")
    vectorstore_file = f"{repo.full_name.replace('/', '_')}_{embeddings_model_name.replace('/', '_')}_vectorstore"
    repo_state_file = f"{repo.full_name.replace('/', '_')}_repo_state.json"
    vectorstore_path = os.path.join(".cache", vectorstore_file)
    repo_state_path = os.path.join(".cache", repo_state_file)

    if os.path.exists(vectorstore_path) and os.path.getsize(repo_state_path) > 0:
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


def update_vectorstore(
    repo,
    embeddings,
    github_token,
    vectorstore_path,
    repo_state_path,
    branch="main",
    recent_period=None,
):
    """
    Update the vector store with new documents if there are new commits.

    Args:
        repo (Repository): The GitHub repository object.
        embeddings (object): The embeddings model object.
        github_token (str): GitHub token for authentication.
        vectorstore_path (str): Path to save the vector store.
        repo_state_path (str): Path to save the repository state.
        branch (str, optional): Branch to fetch documents from. Defaults to "main".
        recent_period (dict, optional): Dictionary with keys 'months' or 'weeks' to filter documents.

    Returns:
        FAISS: The updated vector store object.
    """
    documents = fetch_documents_from_repo(repo, github_token, branch, recent_period)
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings, allow_dangerous_deserialization=True
    )
    vectorstore.add_documents(documents)
    vectorstore.save_local(vectorstore_path)

    # Save the latest commit SHA to track changes
    latest_commit = repo.get_commits()[0]
    repo_state = {"latest_commit_sha": latest_commit.sha}
    with open(repo_state_path, "w") as f:
        json.dump(repo_state, f)

    logging.info(f"Updated and saved vector store for {repo.full_name}")
    return vectorstore


def load_or_update_vectorstore(
    repo, embeddings, github_token, config, branch="main", recent_period=None
):
    """
    Load an existing vector store and update it if necessary, or build a new one.

    Args:
        repo (Repository): The GitHub repository object.
        embeddings (object): The embeddings model object.
        github_token (str): GitHub token for authentication.
        config (dict): Configuration dictionary.
        branch (str, optional): Branch to fetch documents from. Defaults to "main".
        recent_period (dict, optional): Dictionary with keys 'months' or 'weeks' to filter documents.

    Returns:
        FAISS: The loaded and updated or newly built vector store object.
    """
    embeddings_model_name = config.get("embedding_model_name", "default")
    vectorstore_file = f"{repo.full_name.replace('/', '_')}_{embeddings_model_name.replace('/', '_')}_vectorstore"
    repo_state_file = f"{repo.full_name.replace('/', '_')}_repo_state.json"
    vectorstore_path = os.path.join(".cache", vectorstore_file)
    repo_state_path = os.path.join(".cache", repo_state_file)

    if os.path.exists(vectorstore_path) and os.path.getsize(repo_state_path) > 0:
        with open(repo_state_path, "r") as f:
            repo_state = json.load(f)

        latest_commit_sha = repo.get_commits()[0].sha

        if repo_state.get("latest_commit_sha") == latest_commit_sha:
            vectorstore = FAISS.load_local(
                vectorstore_path, embeddings, allow_dangerous_deserialization=True
            )
            logging.info(f"Loaded existing vector store for {repo.full_name}")
            return vectorstore
        else:
            logging.info(
                f"New commits detected for {repo.full_name}. Updating vector store."
            )
            return update_vectorstore(
                repo,
                embeddings,
                github_token,
                vectorstore_path,
                repo_state_path,
                branch,
                recent_period,
            )

    logging.info(
        f"No existing vector store found. Building a new one for {repo.full_name}"
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

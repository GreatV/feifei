# utils.py

import os
import logging
from .vectorstore_utils import load_or_build_vectorstore
from .github_utils import check_and_reply_new_issues
from langchain.chains.retrieval import create_retrieval_chain

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
    )

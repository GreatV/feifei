import logging
from .vectorstore_utils import load_or_update_vectorstore
from .github_utils import check_and_reply_new_issues, check_and_reply_new_discussions
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
import concurrent.futures
import time


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
    enable_compression,
):
    """
    Process a repository by loading or building the vector store and checking for new issues.

    Args:
        repo_full_name (str): Full name of the repository (owner/repo).
        repo_settings (dict): Repository-specific settings.
        embeddings (object): The embeddings model object.
        combine_docs_chain (object): The document combination chain object.
        github_token (str): GitHub token for authentication.
        config (dict): Configuration dictionary.
        args (Namespace): Command-line arguments.
        llm_model_name (str): Name of the language model being used.
        github_client (Github): The GitHub client object.
        signature_template (str): Template for response signatures.
        enable_compression (bool): Whether to enable document compression.

    Returns:
        None
    """
    start_issue_number = repo_settings.get("start_issue_number", 1)
    branch = repo_settings.get("branch", None)
    recent_period = repo_settings.get("recent_period", None)
    repo = github_client.get_repo(repo_full_name.strip())
    logging.info(f"Processing repository: {repo_full_name}")

    # Load or update vector store
    vectorstore = load_or_update_vectorstore(
        repo, embeddings, github_token, config, branch, recent_period
    )

    # Initialize LLMLinguaCompressor for document compression if enabled
    if enable_compression:
        compressor = LLMLinguaCompressor(
            model_name="TheBloke/Llama-2-7b-Chat-GPTQ",
            model_config={"revision": "main"},
            device_map="auto",
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create retrieval chain
    qa_chain = create_retrieval_chain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever,  # Use the appropriate retriever
    )

    # Check and reply to new issues if enabled
    if repo_settings.get("enable_issue_reply", False):
        check_and_reply_new_issues(
            repo,
            retriever,
            qa_chain,
            llm_model_name,
            start_issue_number,
            args.debug,
            signature_template,
        )

    # Check and reply to new discussions if enabled for the repository
    if repo_settings.get("enable_discussion_reply", False):
        start_discussion_number = repo_settings.get("start_discussion_number", 1)
        check_and_reply_new_discussions(
            repo,
            retriever,
            qa_chain,
            llm_model_name,
            start_discussion_number,
            args.debug,
            signature_template,
            github_token,
        )


def process_repositories_batch(repositories_config, batch_size=3, **kwargs):
    """
    Process repositories in batches to avoid resource overuse.

    Args:
        repositories_config (dict): Configuration dictionary for all repositories.
        batch_size (int, optional): Number of repositories to process simultaneously. Defaults to 3.
        **kwargs: Additional keyword arguments to pass to process_repository function.

    Returns:
        None

    Note:
        This function includes a delay between batches to avoid hitting API rate limits.
    """
    repos = list(repositories_config.items())
    for i in range(0, len(repos), batch_size):
        batch = repos[i : i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_repository, repo_full_name, repo_settings, **kwargs
                ): repo_full_name
                for repo_full_name, repo_settings in batch
            }
            for future in concurrent.futures.as_completed(futures):
                repo_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing {repo_name}: {e}")
        time.sleep(5)  # pause between batches to avoid API limit

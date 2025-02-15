import os
import time
import logging
import argparse
import concurrent.futures
from dotenv import load_dotenv
from github import Github
from langchain.chains.combine_documents import create_stuff_documents_chain

from feifei.config import load_config, validate_config
from feifei.llm_utils import initialize_llm, initialize_embeddings, get_prompt_template
from feifei.utils import process_repository

# Load environment variables from a .env file if present
load_dotenv()


# Configure logging
log_level = "INFO"
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("feifei-bot.log"),
        logging.StreamHandler(),
    ],
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
        anthropic_api_base=config.get("anthropic_api_base"),
    )

    # Check if compression is enabled
    enable_compression = config.get("enable_compression", False)

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
                    enable_compression,
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

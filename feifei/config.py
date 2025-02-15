import yaml
import logging
import sys


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
    """configuration validation"""
    required_keys = {
        "github_base_url": str,
        "repositories": dict,
        "embedding_backend": str,
        "llm_backend": str,
        "model_name": str,
        "signature": str,
        "prompt_template": str,
    }

    for key, expected_type in required_keys.items():
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            sys.exit(1)
        if not isinstance(config[key], expected_type):
            logging.error(
                f"Invalid type for {key}: expected {expected_type.__name__}, got {type(config[key]).__name__}"
            )
            sys.exit(1)

    # validate repository configuration
    for repo, repo_config in config["repositories"].items():
        required_repo_keys = {
            "start_issue_number": int,
            "branch": str,
            "recent_period": dict,
            "enable_discussion_reply": bool,
            "enable_issue_reply": bool,
        }
        for key, expected_type in required_repo_keys.items():
            if key not in repo_config:
                logging.warning(
                    f"Missing recommended key '{key}' in repository '{repo}'"
                )
            elif not isinstance(repo_config[key], expected_type):
                logging.error(f"Invalid type for {repo}.{key}")
                sys.exit(1)

    logging.info("Configuration validation passed.")

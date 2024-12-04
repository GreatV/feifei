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

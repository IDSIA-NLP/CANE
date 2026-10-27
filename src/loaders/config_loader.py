import yaml
from loguru import logger

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            logger.debug(config)
            return config
        except yaml.YAMLError as exc:
            logger.error(f"Error loading config: {exc}")
            raise

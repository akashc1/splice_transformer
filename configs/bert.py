from configs import base as base_config
from constants import ModelType


def get_config():
    config = base_config.get_config()
    config.model_type = ModelType.BERT

    return config

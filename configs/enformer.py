
from configs import bert as base_config
from constants import ModelType


def get_config():
    config = base_config.get_config()
    config.model_type = ModelType.ENFORMER
    config.conv_setting = 80

    return config

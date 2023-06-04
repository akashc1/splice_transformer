from configs import base as base_config


def get_config():
    config = base_config.get_config()
    config.context_length = 400

    return config

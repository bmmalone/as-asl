###
#   This module contains various helpers for the AS-autosklearn wrapper
#   project.
###
import logging
logger = logging.getLogger(__name__)

import misc.utils as utils

def load_config(config, required_keys=None):
    """ Read in the config file, print a logging (INFO) statement and verify
    that the required keys are present
    """
    import yaml

    msg = "Reading config file"
    logger.info(msg)

    config = yaml.load(open(config))

    if required_keys is not None:
        utils.check_keys_exist(config, required_keys)
    return config


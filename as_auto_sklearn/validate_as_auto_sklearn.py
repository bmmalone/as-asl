#! /usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import yaml

import misc.pandas_utils as pandas_utils
import misc.parallel as parallel
import misc.utils as utils

from aslib_scenario.aslib_scenario import ASlibScenario
from autofolio.validation.validate import Validator

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def _get_schedule(row, budget):
    """ Return a schedule in the form required by the autofolio validator
    base on the prediction given in the row.
    """

    # {instance name -> (list of) tuples [algo, bugdet]}
    ret = {
        row['instance_id']: [tuple([row['solver'], budget])]
    }
    
    return ret

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Validate the algorithm selection performance of the "
        "predictions made using test-as-auto-sklearn using "
        "autofolio.validation.validate.Validator.")
    
    parser.add_argument('scenario', help="The ASlib scenario")
    parser.add_argument('predictions', help="The predictions file, from "
        "test-as-auto-sklearn")

    parser.add_argument('--config', help="A (yaml) config file which "
        "specifies options controlling the learner behavior")
    
    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    msg = "Loading ASlib scenario"
    logger.info(msg)

    scenario = ASlibScenario()
    scenario.read_scenario(args.scenario)

    if args.config is not None:
        msg = "Loading yaml config file"
        logger.info(msg)
        config = yaml.load(open(args.config))
    else:
        config = {}
        config['allowed_feature_groups'] = [scenario.feature_group_dict.keys()]
    
    # either way, update the scenario with the features used during training
    scenario.used_feature_groups = config['allowed_feature_groups']

    msg = "Reading predictions"
    logger.info(msg)
    predictions = pd.read_csv(args.predictions)

    msg = "Selecting the algorithm with smallest prediction for each instance"
    logger.info(msg)

    algorithm_selections = pandas_utils.get_group_extreme(
        predictions,
        "predicted",
        ex_type="min",
        group_fields="instance_id"
    )

    msg = "Creating the schedules for the validator"
    logger.info(msg)

    schedules = parallel.apply_df_simple(
        algorithm_selections,
        _get_schedule,
        scenario.algorithm_cutoff_time
    )

    schedules = utils.merge_dicts(*schedules)

    val = Validator()
    performance_type = scenario.performance_type[0]

    if performance_type == "runtime":
        stats = val.validate_runtime(
            schedules=schedules, 
            test_scenario=scenario
        )
        
    elif performance_type == "solution_quality":
        stats = val.validate_quality(
            schedules=schedules, 
            test_scenario=scenario
        )
        
    else:
        msg = "Unknown performance type: {}".format(performance_type)
        raise ValueError(msg)

    msg = "=== RESULTS ==="
    logger.info(msg)
    stats.show()

if __name__ == '__main__':
    main()

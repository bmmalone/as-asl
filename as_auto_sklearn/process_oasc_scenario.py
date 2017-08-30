#! /usr/bin/env python3

import argparse
import json

import numpy as np

import misc.automl_utils as automl_utils

import as_auto_sklearn.as_asl_command_line_utils as clu
import as_auto_sklearn.as_asl_utils as as_asl_utils
import as_auto_sklearn.as_asl_filenames as filenames
from as_auto_sklearn.oasc_test_scenario import OascTestScenario

from as_auto_sklearn.as_asl_ensemble import ASaslScheduler

from as_auto_sklearn.validate import Validator

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a model on an ASlib scenario and use it to create "
        "a solver schedule for an OASC test scenario")

    clu.add_config(parser)
    clu.add_oasc_scenario_options(parser)
    clu.add_num_cpus(parser)
    clu.add_cv_options(parser)
    clu.add_scheduler_options(parser)

    automl_utils.add_automl_options(parser)
    automl_utils.add_blas_options(parser)

    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    # check if we need to spawn a new process for blas
    if automl_utils.spawn_for_blas(args):
        return

    # make sure our arguments were valid
    clu.validate_folds_options(args)
    if args.max_feature_steps < 1:
        args.max_feature_steps = np.inf

    if len(args.folds) == 0:
        args.folds = list(range(1,11))

    required_keys = ['base_path']
    config = as_asl_utils.load_config(args.config, required_keys)

    training_scenario = automl_utils.load_scenario(args.training_scenario, return_name=False)
    testing_scenario = OascTestScenario(args.testing_scenario)
    
    as_asl_scheduler = ASaslScheduler(args, config)
    as_asl_scheduler_fit = as_asl_scheduler.fit(training_scenario)

    msg = "Writing the scheduler to disk"
    logger.info(msg)

    
    model_type = "asl.scheduler"
    if args.use_random_forests:
        model_type = "rf.scheduler"

    scheduler_filename = filenames.get_model_filename(
        config['base_path'],
        model_type,
        scenario=testing_scenario.scenario.scenario,
        note=config.get('note')
    )

    as_asl_scheduler_fit.dump(scheduler_filename)

    msg = "Creating a schedule for the test scenario"
    logger.info(msg)
    test_schedule = as_asl_scheduler.create_schedules(testing_scenario.scenario)

    msg = "Writing the schedule to disk"
    logger.info(msg)


    t = "asl.{}".format(testing_scenario.scenario.scenario)
    if args.use_random_forests:
        t = "rf.{}".format(testing_scenario.scenario.scenario)

    schedule_filename = filenames.get_schedule_filename(
        config['base_path'],
        t,
        note=config.get('note')
    )

    with open(schedule_filename, "w") as fp:
        json.dump(test_schedule, fp=fp, indent=2)


if __name__ == '__main__':
    main()

#! /usr/bin/env python3

import argparse

import misc.automl_utils as automl_utils

import as_asl.as_asl_command_line_utils as clu
import as_asl.as_asl_filenames as filenames
import as_asl.as_asl_utils as as_asl_utils

from as_asl.as_asl_ensemble import ASaslScheduler
from as_asl.validate import Validator

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate the models from process-oasc-scenario")

    clu.add_config(parser)
    clu.add_scenario(parser)
    clu.add_cv_options(parser)
    clu.add_scheduler_options(parser)


    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    # see which folds to run
    if len(args.folds) == 0:
        args.folds = list(range(1,11))
    clu.validate_folds_options(args)

    required_keys = ['base_path']
    config = as_asl_utils.load_config(args.config, required_keys)

    msg = "Loading the schenario"
    logger.info(msg)
    scenario = automl_utils.load_scenario(args.scenario, return_name=False)

    msg = "Loading scheduler"
    logger.info(msg)
    
    model_type = "asl.scheduler"
    if args.use_random_forests:
        model_type = "rf.scheduler"

    scheduler_filename = filenames.get_model_filename(
        config['base_path'],
        model_type,
        scenario=scenario.scenario,
        note=config.get('note')
    )

    scheduler = ASaslScheduler.load(scheduler_filename)

    msg = "Creating schedules for training set"
    logger.info(msg)
    schedules = scheduler.create_schedules(scenario)

    msg = "Stats on training set"
    logger.info(msg)

    validator = Validator() 
    training_stats = validator.validate(
        schedules=schedules,
        test_scenario=scenario
    )
    training_stats.show()

    total_oracle_par1 = 0.0
    total_par1 = 0.0
    total_par10 = 0.0
    total_timeouts = 0
    total_solved = 0

    for fold in args.folds:
        msg = "*** Fold {} ***".format(fold)
        logger.info(msg)
        
        testing, training = scenario.get_split(fold)
        
        msg = "Refitting the model"
        logger.info(msg)
        scheduler = scheduler.refit(training)    

        msg = "Creating schedules for the test set"
        logger.info(msg)
        schedules = scheduler.create_schedules(testing)

        validator = Validator() 
        stat = validator.validate(
            schedules=schedules,
            test_scenario=testing,
            show=True
        )
            
        total_oracle_par1 += stat.oracle_par1
        total_par1 += stat.par1
        total_par10 += stat.par10
        total_timeouts += stat.timeouts
        total_solved += stat.solved

    total = total_timeouts + total_solved
    total_oracle_par1 = total_oracle_par1 / total
    total_par10 = total_par10 / total
    total_par1 = total_par1 / total


    msg = "Total (average) Oracle PAR1: {:.4f}".format(total_oracle_par1)
    logger.info(msg)

    msg = "Total (average) PAR10: {:.4f}".format(total_par10)
    logger.info(msg)

    msg = "Total (average) PAR1: {:.4f}".format(total_par1)
    logger.info(msg)

    msg = "Total Timeouts: {} / {}".format(total_timeouts, total)
    logger.info(msg)

    msg = "Total Solved: {} / {}".format(total_solved, total)
    logger.info(msg)

if __name__ == '__main__':
    main()

#! /usr/bin/env python3

import argparse
import itertools
import os

import pandas as pd

import misc.automl_utils as automl_utils
import misc.parallel as parallel

import as_auto_sklearn.as_asl_command_line_utils as clu
import as_auto_sklearn.as_asl_filenames as filenames
import as_auto_sklearn.as_asl_utils as as_asl_utils

from as_auto_sklearn.as_asl_ensemble import ASaslScheduler
from as_auto_sklearn.validate import Validator

import misc.pandas_utils as pd_utils
import misc.utils as utils

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def get_stats_summary(scenario_use_random_forests, args, config):

    scenario, use_random_forests = scenario_use_random_forests
    
    msg = "Loading the scenario"
    logger.info(msg)
    scenario = automl_utils.load_scenario(scenario, return_name=False)

    msg = "Loading scheduler"
    logger.info(msg)
    
    model_type = "asl.scheduler"
    if use_random_forests:
        model_type = "rf.scheduler"

    scheduler_filename = filenames.get_model_filename(
        config['base_path'],
        model_type,
        scenario=scenario.scenario,
        note=config.get('note')
    )
    
    if not os.path.exists(scheduler_filename):
        msg = "Could not find scheduler: {}".format(scheduler_filename)
        logger.warning(msg)
        ret = {
            "scenario": scenario.scenario
        }
        return ret

    scheduler = ASaslScheduler.load(scheduler_filename)

    msg = "Creating schedules for training set"
    logger.info(msg)
    schedules = scheduler.create_schedules(scenario)

    msg = "Stats on training set"
    logger.info(msg)

    validator = Validator() 
    training_stats = validator.validate(
        schedules=schedules,
        test_scenario=scenario,
        show=False
    )
    
    training_stats.total = training_stats.timeouts + training_stats.solved
    training_stats.oracle_par1 = training_stats.oracle_par1 / training_stats.total
    training_stats.par10 = training_stats.par10 / training_stats.total
    training_stats.par1 = training_stats.par1 / training_stats.total
    
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
            show=False
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
    
    ret = {
        "scenario": scenario.scenario,
        "training_oracle_par1":  training_stats.oracle_par1,
        "training_par1":  training_stats.par1,
        "training_par10":  training_stats.par10,
        "training_timeouts":  training_stats.timeouts,
        "training_solved":  training_stats.solved,
        "total_oracle_par1":  total_oracle_par1,
        "total_par1":  total_par1,
        "total_par10":  total_par10,
        "total_timeouts":  total_timeouts,
        "total_solved":  total_solved,
    }

    return ret

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Summarize the evaluation metrics for all scenarios")
    
    clu.add_config(parser)
    parser.add_argument('out')

    clu.add_cv_options(parser)
    clu.add_num_cpus(parser)

    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    # see which folds to run
    if len(args.folds) == 0:
        args.folds = list(range(1,11))
    clu.validate_folds_options(args)

    required_keys = ['base_path', 'training_scenarios_path']
    config = as_asl_utils.load_config(args.config, required_keys)

    scenarios = utils.list_subdirs(config['training_scenarios_path'])
    use_random_forests = [False] #, True]
    it = itertools.product(scenarios, use_random_forests)

    all_stats = parallel.apply_parallel_iter(
        it,
        args.num_cpus,
        get_stats_summary,
        args,
        config
    )

    msg = "Combining statistics"
    logger.info(msg)

    all_stats_df = pd.DataFrame(all_stats)
    pd_utils.write_df(
        all_stats_df,
        args.out,
        create_path=True,
        do_not_compress=True,
        index=False
    )


if __name__ == '__main__':
    main()

#! /usr/bin/env python3

import argparse

import misc.automl_utils as automl_utils
import misc.parallel as parallel

import as_auto_sklearn.as_asl_command_line_utils as clu
import as_auto_sklearn.as_asl_filenames as filenames
import as_auto_sklearn.as_asl_utils as as_asl_utils
from as_auto_sklearn.as_asl_ensemble import ASaslPipeline

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def _log_info(msg, scenario_name, fold):
    msg = "[{}, fold {}]: {}".format(scenario_name, fold, msg)
    logger.info(msg)

def _outer_cv(fold, args, config):
    msg = "loading the scenario"
    _log_info(msg, args.scenario, fold)
    
    scenario_name, scenario = automl_utils.load_scenario(args.scenario)
    
    msg = "extracting fold training data"
    _log_info(msg, scenario_name, fold)
    
    testing, training = scenario.get_split(fold)
    
    msg = "constructing and fitting the pipeline"
    _log_info(msg, scenario_name, fold)
    
    pipeline = ASaslPipeline(args)
    pipeline_fit = pipeline.fit(scenario=training)
    
    msg = "writing pipeline to disk"
    _log_info(msg, scenario_name, fold)
    
    model_type = scenario.scenario
    model_filename = filenames.get_model_filename(
        config['base_path'],
        model_type,
        fold=fold,
        note=config.get('note')
    )

    pipeline_fit.dump(model_filename)
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the Bayesian optimization-based approach for "
        "training models for algorithm selection.")

    clu.add_config(parser)
    clu.add_scenario(parser)
    clu.add_num_cpus(parser)
    clu.add_cv_options(parser)

    automl_utils.add_automl_options(parser, default_total_training_time=20)
    automl_utils.add_blas_options(parser)

    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    # see which folds to run
    if len(args.folds) == 0:
        args.folds = [f for f in range(args.num_folds)]
    clu.validate_folds_options(args)

    required_keys = ['base_path']
    config = as_asl_utils.load_config(args.config, required_keys)

    # check if we need to spawn a new process for blas
    if automl_utils.spawn_for_blas(args):
        return

    pipeline = parallel.apply_parallel_iter(
        args.folds,
        args.num_cpus,
        _outer_cv,
        args,
        config,
        progress_bar=True
    )
if __name__ == '__main__':
    main()

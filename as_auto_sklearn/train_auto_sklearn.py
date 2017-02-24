#! /usr/bin/env python3

import argparse

import joblib
import numpy as np
import pandas as pd
import yaml

import sklearn.cross_validation
import sklearn.pipeline

from autosklearn.regression import AutoSklearnRegressor
from aslib_scenario.aslib_scenario import ASlibScenario

import misc.automl_utils as automl_utils
import misc.math_utils as math_utils
import misc.utils as utils

from misc.column_selector import ColumnSelector

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def _get_pipeline(args, config, scenario):
    pipeline_steps = []

    # find the allowed features
    allowed_features = scenario.features
    allowed_feature_groups = config.get('allowed_feature_groups', [])
    if len(allowed_feature_groups) > 0:
        allowed_features = [
            scenario.feature_group_dict[feature_group]['provides']
                for feature_group in allowed_feature_groups
        ]
        
        allowed_features = utils.flatten_lists(allowed_features)

    feature_set_selector = ColumnSelector(
        allowed_features, 
        transform_contiguous=True
    )

    fs = ('feature_set', feature_set_selector)
    pipeline_steps.append(fs)
    
    # TODO: optionally, we may standardize the data
    #if args.standarize:
    #    s = ('scaler', sklearn.preprocessing.StandardScaler())
    #    pipeline_steps.append(s)

    # then we use the auto-sklearn options
        
    regressor = AutoSklearnRegressor(
        time_left_for_this_task=args.total_training_time,
        per_run_time_limit=args.iteration_time_limit,
        ensemble_size=args.ensemble_size,
        ensemble_nbest=args.ensemble_nbest,
        seed=args.seed,
        include_estimators=args.estimators,
        tmp_folder=args.tmp
    )
    regressor = automl_utils.AutoML(autosklearn_model=regressor)

    r = ('automl', regressor)
    pipeline_steps.append(r)

    # and create the pipeline
    pipeline = sklearn.pipeline.Pipeline(pipeline_steps)

    return pipeline

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script trains models using autosklearn. It then "
        "writes them to disk. Importantly, it *does not* collect any statistics, "
        "make predictions, etc.")

    parser.add_argument('scenario', help="The ASlib scenario")
    
    parser.add_argument('solver', help="The solver for which models will be "
        "learned")

    parser.add_argument('fold', help="The fold for which a model wil be learned",
        type=int)

    parser.add_argument('out', help="The output (pickle) file")

    parser.add_argument('--config', help="A (yaml) config file which specifies "
        "options controlling the learner behavior")

    automl_utils.add_automl_options(parser)
    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    math_utils.check_range(args.fold, 1, 10, variable_name="fold")

    if args.config is not None:
        msg = "Reading config file"
        logger.info(msg)
        config = yaml.load(open(args.config))
    else:
        config = {}

    msg = "Reading ASlib scenario"
    logger.info(msg)
    scenario = ASlibScenario()
    scenario.read_scenario(args.scenario)

    # ensure the selected solver is present
    if args.solver not in scenario.algorithms:
        msg = ("[train-auto-sklear]: the solver '{}' is not present in the "
            "ASlib scenario".format(args.solver))
        raise ValueError(msg)
    
    msg = "Solver: {}, Fold: {}".format(args.solver, args.fold)
    logger.info(msg)

    msg = "Constructing pipeline"
    logger.info(msg)
    pipeline = _get_pipeline(args, config, scenario)

    msg = "Extracting solver and fold performance data"
    logger.info(msg)
    testing, training = scenario.get_split(args.fold)

    X_train = training.feature_data
    
    y_train = training.performance_data[args.solver].values
    #y_train=np.ascontiguousarray(y_train)

    # fit the pipeline on X_train and y_train
    pl_fit = pipeline.fit(X_train, y_train)

    for i in pl_fit.steps:
        print(i)

    msg = "Writing auto-sklearn ensemble to disk: {}".format(args.out)
    logger.info(msg)
    #automl = pl_fit.named_steps['automl']
    #automl_utils.write_automl(automl, args.out)

    joblib.dump(pl_fit, args.out)

if __name__ == '__main__':
    main()

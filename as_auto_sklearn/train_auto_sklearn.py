#! /usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

import sklearn.cross_validation
import sklearn.pipeline

from autosklearn.regression import AutoSklearnRegressor

import misc.automl_utils as automl_utils
import misc.utils as utils

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

#from bn_portfolio.bn_portfolio_utils import ColumnSelector, fields_mapping, solvers, prediction_fields

import bn_portfolio.bn_portfolio_utils as bn_portfolio_utils
import bn_portfolio.bn_portfolio_filenames as filenames

feature_set_choices = bn_portfolio_utils.fields_mapping.keys()

def get_pipeline(args):
    pipeline_steps = []

    # figure out what our feature set looks like
    feature_set_fields = np.array(bn_portfolio_utils.fields_mapping[args.feature_set])

    # the first step in the pipeline is filtering the forbidden features
    feature_set_selector = bn_portfolio_utils.ColumnSelector(feature_set_fields, 
        transform_contiguous=True)

    fs = ('feature_set', feature_set_selector)
    pipeline_steps.append(fs)
    
    # optionally, we may standardize the data
    if args.standarize:
        s = ('scaler', sklearn.preprocessing.StandardScaler())
        pipeline_steps.append(s)

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

    parser.add_argument('instances', help="The preprocessed data")
    
    parser.add_argument('feature_set', help="The set of features to use for this "
        "experiment", choices=feature_set_choices)

    parser.add_argument('solver', help="The solver for which models will be "
        "learned", choices=bn_portfolio_utils.solvers)

    parser.add_argument('fold', help="The fold for which a model wil be learned",
        type=int)

    parser.add_argument('name', help="The base name for the output pickle files")

    parser.add_argument('--standardize', help="If this flag is given, the data "
        "will be standardized (zero mean, unit variance) before learning. N.B. "
        "This script *does not* write the standardized scaler to disk. It must "
        "be recomputed at test time.", action='store_true')

    automl_utils.add_automl_options(parser)
    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    msg = "Reading in the preprocessed data"
    logger.info(msg)
    instances = pd.read_csv(args.instances)

    msg = "Constructing pipeline"
    logger.info(msg)
    pipeline = get_pipeline(args)

    msg = "Removing fields with null values"
    logger.info(msg)

    for f in bn_portfolio_utils.fields_mapping[args.feature_set]:
        m_nan = instances[f].isnull()
        if sum(m_nan) > 0:
            instances = instances[~m_nan]

    msg = "Splitting data into 10-folds"
    logger.info(msg)

    cv = sklearn.cross_validation.KFold(len(instances), n_folds=10, shuffle=True, 
        random_state=args.seed)

    solver_runtimes = instances[args.solver]
    prediction_field = "{}_prediction".format(args.solver)
    error_field = "{}_error".format(args.solver)

    instances[prediction_field] = 0
    train, test = utils.nth(cv, args.fold)

    msg = "Solver: {}, Fold: {}".format(args.solver, args.fold)
    logger.info(msg)

    # note, these create COPIES NOT VIEWS
    X_train, X_test = instances.iloc[train], instances.iloc[test], 
    y_train, y_test = solver_runtimes.iloc[train], solver_runtimes.iloc[test]
    
    # must be numpy for automl
    y_train = y_train.values

    # fit the pipeline on X_train and y_train
    pl_fit = pipeline.fit(X_train, y_train)

    automl = pl_fit.named_steps['automl']

    fn = filenames.aml_pkl(args.out, args.name, args.solver, args.feature_set, 
        args.fold)
    msg = "Writing ensemble to disk: {}".format(fn)
    logger.info(msg)
    automl_utils.write_automl(automl, fn)

if __name__ == '__main__':
    main()

#! /usr/bin/env python3

import argparse
import ctypes
import itertools
import joblib
import numpy as np
import os
import pandas as pd
import shlex
import string
import sys
import yaml

import sklearn.pipeline
import sklearn.preprocessing

from aslib_scenario.aslib_scenario import ASlibScenario

import misc.automl_utils as automl_utils
import misc.math_utils as math_utils
import misc.parallel as parallel
import misc.shell_utils as shell_utils
import misc.utils as utils

from misc.column_selector import ColumnSelector
from misc.column_selector import ColumnTransformer

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

default_num_cpus = 1
default_num_blas_cpus = 1

def _get_pipeline(args, config, scenario):
    """ Create the pipeline which will later be trained to predict runtimes.

    Parameters
    ----------
    args: argparse.Namespace
        The options for training the autosklearn regressor

    config: dict-like
        Other configuration options, such as whether to preprocess the data

    scenario: ASlibScenario
        The scenario. *N.B.* This is only used to get the feature names and
        grouping. No information is "leaked" to the pipeline.
    """
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

    feature_set_selector = ColumnSelector(allowed_features)

    fs = ('feature_set', feature_set_selector)
    pipeline_steps.append(fs)
    
    # check for taking the log
    if 'fields_to_log' in config:
        # then make sure all of the fields are in the scenario
        fields_to_log = config['fields_to_log']
        missing_fields = [f for f in fields_to_log if f not in scenario.features]
        if len(missing_fields) > 0:
            missing_fields_str = ",".join(missing_fields)
            msg = ("[train-auto-sklearn]: Could not find the following fields "
                "to log: {}".format(missing_fields_str))
            raise ValueError(msg)

        log_transformer = ColumnTransformer(fields_to_log, np.log1p)
        log_transformer = ('log_transformer', log_transformer)
        pipeline_steps.append(log_transformer)

    # handling missing values
    imputer_strategy = config.get('imputer_strategy', 'mean')
    automl_utils.check_imputer_strategy(
        imputer_strategy,
        raise_error=True,
        error_prefix="[train-auto-sklearn]: "
    )

    msg = ("[train-auto-sklearn]: Using imputation strategy: {}".format(
        imputer_strategy))
    logger.debug(msg)

    imputer = sklearn.preprocessing.Imputer(strategy=imputer_strategy)
    imputer = ('imputer', imputer)
    pipeline_steps.append(imputer)

    # optionally, check if we want to preprocess
    preprocessing_strategy = config.get('preprocessing_strategy', None)
    if preprocessing_strategy == 'scale':
        msg = ("[train-auto-sklearn]: Adding standard scaler (zero mean, unit "
            "variance) for preprocessing")
        logger.debug(msg)

        s = ('scaler', sklearn.preprocessing.StandardScaler())
        pipeline_steps.append(s)
    elif preprocessing_strategy is None:
        # no preprocessing is fine
        pass
    else:
        msg = ("[train-auto-sklearn]: The preprocessing strategy is not "
            "recognized. given: {}".format(preprocessing_strategy))
        raise ValueError(msg)

    # last, we need to convert it to a "contiguous" array
    ct = sklearn.preprocessing.FunctionTransformer(np.ascontiguousarray)
    ct = ("contiguous_transformer", ct)
    pipeline_steps.append(ct)

    # then we use the auto-sklearn options

    # in order to match with AutoFolio, check if we have wallclock_limit in the
    # config file
    args.total_training_time = config.get("wallclock_limit", 
        args.total_training_time)

    regressor = automl_utils.AutoSklearnWrapper()
    regressor.create_regressor(args)
    r = ('auto_sklearn', regressor)
    pipeline_steps.append(r)

    # and create the pipeline
    pipeline = sklearn.pipeline.Pipeline(pipeline_steps)

    return pipeline

def _outer_cv(solver_fold, args, config):

    solver, fold = solver_fold

    # there are problems serializing the aslib scenario, so just read it again
    scenario = ASlibScenario()
    scenario.read_scenario(args.scenario)
     
    msg = "Solver: {}, Fold: {}".format(solver, fold)
    logger.info(msg)

    msg = "Constructing template pipeline"
    logger.info(msg)
    pipeline = _get_pipeline(args, config, scenario)

    msg = "Extracting solver and fold performance data"
    logger.info(msg)
    
    testing, training = scenario.get_split(fold)
    X_train = training.feature_data
    y_train = training.performance_data[solver].values

    if 'log_performance_data' in config:
        y_train = np.log1p(y_train)
    
    msg = "Fitting the pipeline"
    logger.info(msg)
    pipeline = pipeline.fit(X_train, y_train)

    out = string.Template(args.out)
    out = out.substitute(solver=solver, fold=fold)

    msg = "Writing fit pipeline to disk: {}".format(out)
    logger.info(msg)
    joblib.dump(pipeline, out)

    return pipeline


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script trains a model to predict the runtime for a "
        "solver from an ASlib scenario using autosklearn. It assumes an "
        "\"outer\" cross-validation strategy, and it only trains a model for "
        "the indicated folds and solvers. It then writes the learned model to "
        "disk. It *does not* collect any statistics, make predictions ,etc.")

    parser.add_argument('scenario', help="The ASlib scenario")
    
    parser.add_argument('out', help="A template string for the filenames for "
        "the learned models. They are written with joblib.dump, so they need "
        "to be read back in with joblib.load. ${solver} and ${fold} are the "
        "template part of the string. It is probably necessary to surround "
        "this argument with single quotes in order to prevent shell "
        "replacement of the template parts.")

    parser.add_argument('--config', help="A (yaml) config file which specifies "
        "options controlling the learner behavior")

    parser.add_argument('--solvers', help="The solvers for which models will "
        "be learned. By default, models for all solvers are learned", 
        nargs='*', default=[])

    parser.add_argument('--folds', help="The outer-cv folds for which a model "
        "will be learned. By default, models for all folds are learned", 
        type=int, nargs='*', default=[])

    parser.add_argument('-p', '--num-cpus', help="The number of CPUs to use "
        "for parallel solver/fold training", type=int, 
        default=default_num_cpus)
    
    parser.add_argument('--num-blas-threads', help="The number of threads to "
        "use for parallelizing BLAS. The total number of CPUs will be "
        "\"num_cpus * num_blas_cpus\". Currently, this flag only affects "
        "OpenBLAS and MKL.", type=int, default=default_num_blas_cpus)

    parser.add_argument('--do-not-update-env', help="By default, num-blas-threads "
        "requires that relevant environment variables are updated. Likewise, "
        "if num-cpus is greater than one, it is necessary to turn off python "
        "assertions due to an issue with multiprocessing. If this flag is "
        "present, then the script assumes those updates are already handled. "
        "Otherwise, the relevant environment variables are set, and a new "
        "processes is spawned with this flag and otherwise the same "
        "arguments. This flag is not inended for external users.",
        action='store_true')

    automl_utils.add_automl_options(parser)
    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    # see which folds to run
    folds = args.folds
    if len(folds) == 0:
        folds = range(1, 11)

    for f in folds:
        math_utils.check_range(f, 1, 10, variable_name="fold")

    # and which solvers
    msg = "Reading ASlib scenario"
    logger.info(msg)
    scenario = ASlibScenario()
    scenario.read_scenario(args.scenario)

    # ensure the selected solver is present
    solvers = args.solvers
    if len(solvers) == 0:
        solvers = scenario.algorithms

    for solver in solvers:
        if solver not in scenario.algorithms:
            solver_str = ','.join(scenario.algorithms)
            msg = ("[train-auto-sklear]: the solver is not present in the "
                "ASlib scenario. given: {}. choices: {}".format(solver, 
                solver_str))
            raise ValueError(msg)

    if args.config is not None:
        msg = "Reading config file"
        logger.info(msg)
        config = yaml.load(open(args.config))
    else:
        config = {}

    # everything is present, so update the environment variables and spawn a
    # new process, if necessary
    if not args.do_not_update_env:
        ###
        #
        # There is a lot going on with settings these environment variables.
        # please see the following references:
        #
        #   Turning off assertions so we can parallelize sklearn across
        #   multiple CPUs for different solvers/folds
        #       https://github.com/celery/celery/issues/1709
        #
        #   Controlling OpenBLAS threads
        #       https://github.com/automl/auto-sklearn/issues/166
        #
        #   Other environment variables controlling thread usage
        #       http://stackoverflow.com/questions/30791550
        #
        ###
        
        # we only need to turn off the assertions if we parallelize across cpus
        if args.num_cpus > 1:
            os.environ['PYTHONOPTIMIZE'] = "1"

        # openblas
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_blas_threads)
        
        # mkl blas
        os.environ['MKL_NUM_THREADS'] = str(args.num_blas_threads)

        # other stuff from the SO post
        os.environ['OMP_NUM_THREADS'] = str(args.num_blas_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_blas_threads)

        cmd = ' '.join(shlex.quote(a) for a in sys.argv)
        cmd += " --do-not-update-env"
        shell_utils.check_call(cmd)
        return

    msg = "Learning regressors"
    logger.info(msg)

    it = itertools.product(solvers, folds)
    regressors = parallel.apply_parallel_iter(
        it,
        args.num_cpus,
        _outer_cv,
        args,
        config,
        progress_bar=True
    )
    
if __name__ == '__main__':
    main()

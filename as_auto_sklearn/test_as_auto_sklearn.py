#! /usr/bin/env python3

import argparse
import itertools
import joblib
import numpy as np
import os
import pandas as pd
import string
import yaml

from aslib_scenario.aslib_scenario import ASlibScenario
import misc.automl_utils as automl_utils
import misc.utils as utils

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Test models learned with train-as-auto-sklearn. It "
        "writes the predictions to disk as a \"long\" data frame. The output "
        "file is in gzipped csv format.")
    
    parser.add_argument('scenario', help="The ASlib scenario")
    
    parser.add_argument('model_template', help="A template string for the filenames for "
        "the learned models. ${solver} and ${fold} are the template part of "
        "the string. It is probably necessary to surround this argument with "
        "single quotes in order to prevent shell replacement of the template "
        "parts.")

    parser.add_argument('out', help="The output csv file")

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

    msg = "Creating string templates"
    logger.info(msg)
    model_template = string.Template(args.model_template)

    msg = "Finding folds from ASlib scenario"
    logger.info(msg)
        
    folds = [int(i) for i in scenario.cv_data['fold'].unique()]
    folds = sorted(folds)

    msg = "Making predictions"
    logger.info(msg)

    all_predictions = []
    it = itertools.product(scenario.algorithms, folds)
    for solver, fold in it:
        
        model_file = model_template.substitute(solver=solver, fold=fold)
        
        if not os.path.exists(model_file):
            msg = "Could not find model file. Skipping: {}".format(model_file)
            logger.warning(msg)
            continue
        
        try:
            model = joblib.load(model_file)
        except:
            msg = ("Problem loading the model file. Skipping: {}".format(
                model_file))
            logger.warning(msg)
            continue
            
        msg = "Processing. solver: {}. fold: {}".format(solver, fold)
        logger.info(msg)
        
        testing, training = scenario.get_split(fold)
        y_pred = model.predict(testing.feature_data)

        if 'log_performance_data':
            # exp transform it back out
            y_pred = np.expm1(y_pred)
            
        pred_df = pd.DataFrame()
        pred_df['instance_id'] = testing.feature_data.index
        pred_df['solver'] = solver
        pred_df['fold'] = fold
        pred_df['actual'] = testing.performance_data[solver].values
        pred_df['predicted'] = y_pred
        
        all_predictions.append(pred_df)
        
    msg = "Joining all predictions in a long data frame"
    logger.info(msg)
    all_predictions = pd.concat(all_predictions)

    msg = "Writing predictions to disk"
    logger.info(msg)

    utils.write_df(all_predictions, args.out, index=False)

if __name__ == '__main__':
    main()

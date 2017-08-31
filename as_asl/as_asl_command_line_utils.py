###
#   This module contains helpers to ensure consistency across the command line
#   parameters for the Algorithm Selection auto-sklearn wrapper project.
###

import misc.math_utils as math_utils
import numpy as np

###
# C
###

def add_config(parser):
    """ Add the (required) config parameter to the parser
    """    
    parser.add_argument('config', help="The yaml configutration file")

def get_config_options_string(args):
    """ Create a string suitable for passing to another script with the
    config options
    """
    s = args.config
    return s

def add_cv_options(parser, default_folds=[]):
    """ Add the optional cross-validation parameters
    """
    cv_options = parser.add_argument_group("cross-validation options")
    
    cv_options.add_argument('--folds', help="The outer-cv folds which will be "
        "processed by this script. By default, all folds will be processed. "
        "\n\nN.B. This is based on the ASlibScenario folds, so it is base-1.", 
        type=int, nargs='*', default=default_folds)

def validate_folds_options(args):
    """ Ensure that all specified folds are >= 1 and <= 10
    """
    import misc.math_utils as math_utils
    # make sure they are all valid
    for f in args.folds:
        math_utils.check_range(f, 1, 10, variable_name="fold", 
            max_inclusive=True)

def get_cv_options_string(args):
    """ Create a string suitable for passing to another script with the
    cross-validation options
    """
    args_dict = vars(args)

    s = ""
    if ('folds' in args_dict) and (len(args.folds) > 0):
        s = " ".join(str(f) for f in args.folds)
        s = "--folds {}".format(s)

    return s

###
# N
###

def add_num_cpus(parser, default=1):
    """ Add the optional num_cpus parameter with the specified default
    """
    parser.add_argument('--num-cpus', help="The number of CPUs to use",
        type=int, default=default)

def get_num_cpus_options_string(args):
    args_dict = vars(args)

    s = ""
    if 'num_cpus' in args_dict:
        s = "--num-cpus {}".format(args.num_cpus)
    return s

###
# O
###
def add_oasc_scenario_options(parser):
    """ Add the required training_scenario and testing_scenario options
    """
    parser.add_argument('training_scenario', help="A training ASlib scenario")

    parser.add_argument('testing_scenario', help="A testing scenario from the "
        "OASC, which presumably matches the training scenario")


###
# Presolve
###

def add_simple_presolver_options(parser, default_budget=0.05,
        default_min_fast_solutions=0.5):
    """ Add optional parameters controlling the simple presolver
    """
    presolver_options = parser.add_argument_group("simple presolver options")

    presolver_options.add_argument('--presolver-budget', help="The fraction of the "
        "scenario cutoff time used for presolving", type=float,
        default=default_budget)

    presolver_options.add_argument('--presolver-min-fast-solutions', help="The fraction "
        "of instances which must be solved within the presolver budget to "
        "consider the solver for use as a presolver", type=float,
        default=default_min_fast_solutions)

def validate_simple_presolver_options(args):
    """ Ensure the presolver options are within valid bounds
    """
    math_utils.check_range(args.presolver_budget, 0, 1,
        variable_name='--presolver-budget')

    math_utils.check_range(args.presolver_min_fast_solutions, 0, 1,
        variable_name='--presolver-min-fast-solutions')

def add_enable_presolving_option(parser):
    """ Add the --enable-presolving flag
    """
    parser.add_argument('--enable-presolving', help="If this flag is present, "
        "then presolving will be enabled", action='store_true')


###
# R
###
def add_result_type(parser, default="result-type"):
    """ Add the optional result_type parameter with the specified default
    """
    parser.add_argument('--result-type', help="The identifier for the "
        "results, such as \"bo-baseline\"", default=default)

###
# S
###
def add_scenario(parser, optional=False, default=None):
    """ Add the scenario parameter to the parser

    Parameters
    ----------
    parser: argparse.ArgumentParser
        The parser

    optional: bool
        Whether the --scenario parameter is optional

    default: string
        If the parameter is optional, it will have this default value
    """
    scenario_help = "The ASlibScenario to process"
    if optional:
        parser.add_argument('--scenario', help=scenario_help, default=default)
    else:
        parser.add_argument('scenario', help=scenario_help)

def add_scheduler_options(parser, default_max_feature_steps=0):
    """ Add the --use-random-forests and --max-feature-steps options
    """
    scheduler_options = parser.add_argument_group("scheduler options")

    scheduler_options.add_argument('--use-random-forests', help="If this flag "
        "is present, then only random forest regressors and classifiers, "
        "rather than ensembles learned with auto-sklearn, will be used in the "
        "ensemble.", action='store_true')

    scheduler_options.add_argument('--max-feature-steps', help="The maximum "
        "number of feature steps to select. If this number is less than 1, "
        "then no limit will be used.", type=int,
        default=default_max_feature_steps)

def get_scheduler_options_string(args):
    args_dict = vars(args)

    rf = ""
    fs = ""

    if args.use_random_forests:
        rf = "--use-random-forests"

    if 'max_feature_steps' in args_dict:
        fs = "--max-feature-steps {}".format(args.max_feature_steps)

    s = " ".join([rf, fs])
    return s

def add_seed(parser, default=8675309):
    """ Add a random seed parameter to the parser
    """
    parser.add_argument('--seed', help="The random seed", type=int,
        default=default)


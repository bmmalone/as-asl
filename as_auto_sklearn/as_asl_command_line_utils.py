###
#   This module contains helpers to ensure consistency across the command line
#   parameters for the Algorithm Selection auto-sklearn wrapper project.
###

import misc.math_utils as math_utils

###
# C
###

def add_config(parser):
    """ Add the (required) config parameter to the parser
    """    
    parser.add_argument('config', help="The yaml configutration file")

def add_cv_options(parser, default_num_folds=10, default_folds=[]):
    """ Add the optional cross-validation parameters
    """
    cv_options = parser.add_argument_group("cross-validation options")

    cv_options.add_argument('--num-folds', help="The number of folds to use "
        "during cross-validation", type=int, default=default_num_folds)
    
    cv_options.add_argument('--folds', help="The outer-cv folds which will be "
        "processed by this script. By default, all folds will be processed. "
        "\n\nN.B. This is based on the ASlibScenario folds, so it is base-1.", 
        type=int, nargs='*', default=default_folds)

def validate_folds_options(args):
    """ Ensure that all specified folds are >=0 and < num_folds
    """
    import misc.math_utils as math_utils
    # make sure they are all valid
    for f in args.folds:
        math_utils.check_range(f, 1, args.num_folds, variable_name="fold", 
            max_inclusive=True)


###
# N
###

def add_num_cpus(parser, default=1):
    """ Add the optional num_cpus parameter with the specified default
    """
    parser.add_argument('--num-cpus', help="The number of CPUs to use",
        type=int, default=default)
###
# Presolve
###

def add_simple_presolver_options(parser, default_budget=0.05,
        default_min_fast_solutions=0.5):
    """ Add optional parameters controlling the simple presolver
    """
    parser.add_argument('--presolver-budget', help="The fraction of the "
        "scenario cutoff time used for presolving", type=float,
        default=default_budget)

    parser.add_argument('--presolver-min-fast-solutions', help="The fraction "
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

def add_seed(parser, default=8675309):
    """ Add a random seed parameter to the parser
    """
    parser.add_argument('--seed', help="The random seed", type=int,
        default=default)


###
#   This module contains helpers to ensure consistency across the command line
#   parameters for the Algorithm Selection auto-sklearn wrapper project.
###


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
        math_utils.check_range(f, 0, args.num_folds, variable_name="fold", 
            max_inclusive=False)


###
# N
###

def add_num_cpus(parser, default=1):
    """ Add the optional num_cpus parameter with the specified default
    """
    parser.add_argument('--num-cpus', help="The number of CPUs to use",
        type=int, default=default)

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

def add_seed(parser, default=8675309):
    """ Add a random seed parameter to the parser
    """
    parser.add_argument('--seed', help="The random seed", type=int,
        default=default)


###
#   This module contains functions to construct filenames for the as_asl
#   library.
###

import os


def _get_compression_str(compression_type=None):
    compression_str = ""
    if (compression_type is not None) and  (len(compression_type) > 0):
        compression_str = ".{}".format(compression_type)
    return compression_str

def _get_note_str(note=None):
    note_str = ""
    if (note is not None) and  (len(note) > 0):
        note_str = ".{}".format(note)
    return note_str

def _get_fold_str(fold=None):
    fold_str = ""
    if (fold is not None) and (len(str(fold)) > 0):
        fold_str = ".fold-{}".format(fold)
    return fold_str

###
# M
###
def get_model_filename(base_path, model_type, fold=None,
        compression_type="gz", note=None):
    """ Get the path to a file containing a fit model

    Parameters
    ----------
    base_path: path-like (e.g., a string)
        The path to the base data directory

    model_type: string
        The identifier for the type of model (e.g., "bo-baseline")

    fold: int-like
        The cross-validation fold

    compression_type: string
        The extension for the type of compression. Please see the joblib docs
        for more details about supported compression types and extensions.

        Use None for no compression.

    note: string or None
        Any additional note to include in the filename. This should probably
        not contain spaces or special characters.

    Returns
    -------
    result_filename: string
        The path to the result file
    """

    fname = [
        "model.",
        model_type,
        _get_fold_str(fold),
        _get_note_str(note),
        ".pkl",
        _get_compression_str(compression_type)
    ]

    fname = ''.join(fname)
    fname = os.path.join(base_path, fname)
    return fname


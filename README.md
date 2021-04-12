# `as-asl`

### Deprecated

This project is no longer maintained. However, please feel free to contact me if you have questions about the associated paper or believe parts of the source code may be of use to you.

## Overview

This project incorporates the auto-sklearn toolkit (`asl`) into a solver runtime 
prediction framework. The predictions are then passed to a second classification
model which yields a solution to the algorithm selection problem (`as`).

## Installation

This project is written in `python3` and can be installed with `pip`. 
The project indirectly depends on the [`pyrfr`](https://github.com/automl/random_forest_run)
and, thus, also requires [`SWIG-3`](http://www.swig.org/).

```
pip3 install -r requirements.txt
pip3 install -r requirements.2.txt
```

**N.B.** If installing under anaconda, use `pip` rather than `pip3`

## Example usage for [OASC](http://www.coseal.net/open-algorithm-selection-challenge-2017-oasc/)

All usage requires a yaml configuration file. This needs to include a
`base_path` key, and all results will be to that location.

An example (complete) config file:

```
base_path: /prj/oasc2017
```

### Training models with `auto-sklearn`

**Short test run**
```
process-oasc-scenario oasc.yaml /mldb/oasc_scenarios/train/Bado/ /mldb/oasc_scenarios/test/Bado/ --total-training-time 30 --num-cpus 3 --logging-level INFO --max-feature-steps 1
```

**Command for OASC-like run**
```
process-oasc-scenario oasc.yaml /path/to/oasc_scenarios/train/Bado/ /path/to/oasc_scenarios/test/Bado/ --total-training-time 600 --num-cpus 8 --logging-level INFO
```

This command uses the training set to learn an algorithm selection scheduler. It
then uses that scheduler to create solving schedules for the test set. The
`--total-training-time` parameter gives the approximate amount of time (in
seconds) allocated to `auto-sklearn` for each internal model.

This script also performs feature set selection and determines a presolving
schedule. The `--max-feature-steps` can be given to limit the number of feature
steps considered in the search. 

Typical problem sets for OASC take about an hour or two with 8 CPUs
when 10 minutes are used for each model.

This script also accepts other optional parameters controlling the behavior of
`auto-sklearn`, BLAS, logging, etc. These were all kept at default values for
submissions.

The `--help` flag can be given to see all options and their default values.

The schedule for the test instances is written to: `<base_path>/schedule.asl.<scenario_name>.json`.
The learned scheduler is written to: `<base_path>/model.asl.scheduler.<scenario_name>.pkl.gz`

### Training models with random forests

```
process-oasc-scenario oasc.yaml /path/to/oasc_scenarios/train/Bado/ /path/to/oasc_scenarios/test/Bado/ --use-random-forests --num-cpus 8 --logging-level INFO
```

This command differs from the command above because it includes the
`--use-random-forests` flag. Rather than learn the internal models for the
scheduler using `auto-sklearn`, it instead uses standard `sklearn` random
forests (with 100 trees).

It ignores the `auto-sklearn` parameters (such as `--total-training-time`). It
tends to be somewhat faster since it avoids the Bayesian optimization. Still,
many of the steps (such as feature step selection) are the same for both, so the
typical time on OASC is similar to that mentioned above.

The schedule for the test instances is written to: `<base_path>/schedule.rf.<scenario_name>.json`.
The learned scheduler is written to: `<base_path>/model.rf.scheduler.<scenario_name>.pkl.gz`

### `automl_utils`

This project relies heavily on the [`automl_utils`](https://github.com/bmmalone/pymisc-utils/blob/master/misc/automl_utils.py)
module. Indeed, almost all of the logic for interacting with `auto-sklearn` is
wrapped in that module.

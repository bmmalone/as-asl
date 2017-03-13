# Change Log
All notable changes to as-auto-sklearn will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/), 
and this project adheres to [Semantic Versioning](http://semver.org/).

## Future release - [1.2.0]

The `aspeed` presolver is used by 
[AutoFolio](https://github.com/mlindauer/AutoFolio) to improve algorithm
selection behavior; it can be used by `as-auto-sklearn` in a straight-forward
manner to improve algorithm selection behavior.

## In progress - [1.1.0]

The 1.1.0 release will ensure prediction behavior matches that of the
`bn-portfolio` code used in the paper. The new features will include:

* Distribute learning across a cluster using password-less SSH (based on public
  keys)

Additionally, the release will ensure predictions do not differ "substantially"
from those in the paper. Clearly, this is subjective; nevertheless, the decision
to call the milestone "reached" will be made in good faith.

## [1.0.1] - 2017-03-13

This release includes only updates to documentation. It does not include any
new features, changes of interface, etc.

## [1.0.0] - 2017-03-12
The initial import of code from the (private) bn-portfolio repo, while removing
the BNSL-specific dependencies. In particular, the package now works with
arbitrary ASlib scenarios; if offers the following options:

* Limit the runtime of learning models for individual (solver, fold) pairs.

* Forbid specified feature groups.

* Impute missing values based on simple replacement strategies (like "mean").

* Preprocess the data using standard strategies which are not considered by
  auto-sklearn.

    * Normalize (all) features to have zero mean and unit variance
    * Take the logarithm of specified features and the performance data

* Restrict the computational resources (CPUs and threads) required by the
  pipeline with command line options to the main `train`ining script.

* Predict the runtime of all solvers on all instances.
* Evaluate the algorithm selection performance based on the runtime predictions.

## [0.0.1] - 2017-02-23
The initial commit. This simply establishes the directory structure.
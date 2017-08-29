import logging
logger = logging.getLogger(__name__)

import autosklearn.metrics

import joblib
import numpy as np
import pandas as pd

import mlxtend.feature_selection
import networkx as nx
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.utils.validation import check_is_fitted

import as_auto_sklearn.as_asl_utils as as_asl_utils
import as_auto_sklearn.as_asl_filenames as filenames
from as_auto_sklearn.validate import Validator

import misc.automl_utils as automl_utils
from misc.nan_standard_scaler import NaNStandardScaler
import misc.parallel as parallel
import misc.utils as utils

class ASaslEnsemble:
    def __init__(self, args, solvers, use_random_forests=False):
        self.args = args
        self.solvers = solvers
        self.use_random_forests = use_random_forests
        
    # train each of the regressors
    def _fit_regressor(self, solver):
        model = self.solver_asl_regressors[solver]
        y_train = self.y_train[solver]
        
        # we want to punish large errors, so use mse
        metric = autosklearn.metrics.mean_squared_error

        num_nan = np.isnan(self.X_train).sum()
        num_inf = np.isinf(self.X_train).sum()

        msg = ("[as_asl_ensemble._fit_regressor]: num_nan(X_train): {}".
            format(num_nan))
        logger.debug(msg)
        
        msg = ("[as_asl_ensemble._fit_regressor]: num_inf(X_train): {}".
            format(num_inf))
        logger.debug(msg)

        if self.use_random_forests:
            model_fit = model.fit(self.X_train, y_train)
        else:
            model_fit = model.fit(self.X_train, y_train, metric=metric)
        return (solver, model_fit)


    def _fit_init(self, X_train, y_train):
        
        # make sure we can encode our algorithm labels
        self.le = sklearn.preprocessing.LabelEncoder()
        self.le_ = self.le.fit(self.solvers)
        
        # create the solver-specific datasets
        self.y_train = {
            solver: y_train[solver]
                for solver in self.solvers
        }
        
        # and save the training dataset
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.values
        else:
            self.X_train = X_train
        
        self.orig_y_train = y_train
        
        return self

    
    def _fit_regressors(self):
        
        # create the regressors for each solver

        if self.use_random_forests:
            self.solver_asl_regressors = {
                solver: RandomForestRegressor(
                    n_estimators=100
                ) for solver in self.solvers
            }
        else:
            self.solver_asl_regressors = {
                solver: automl_utils.AutoSklearnWrapper(
                    estimator_named_step="regressor", args=self.args
                ) for solver in self.solvers
            }

        # fit the regressors
        ret = parallel.apply_parallel_iter(
            self.solvers,
            self.args.num_cpus,
            self._fit_regressor
        )

        self.solver_asl_regressors_ = dict(ret)

        return self

    def _get_stacking_model_dataset_asl(self, X):

        X_stacking_train = pd.DataFrame()
        for solver in self.solvers:
            r = self.solver_asl_regressors_[solver]
            mean, std = r.predict_dist(X)
            m = "{}_mean".format(solver)
            s = "{}_std".format(solver)
            X_stacking_train[m] = mean
            X_stacking_train[s] = std

        X = [X, X_stacking_train]
        X = np.concatenate(X, axis=1)
        return X


    def _get_stacking_model_dataset_rf(self, X):

        X_stacking_train = pd.DataFrame()
        for solver in self.solvers:
            r = self.solver_asl_regressors_[solver]
            mean = r.predict(X)
            m = "{}_mean".format(solver)
            X_stacking_train[m] = mean

        X = [X, X_stacking_train]
        X = np.concatenate(X, axis=1)
        return X


        
    def _fit_stacking_model(self, metric=autosklearn.metrics.f1_micro):
        
        # use those to build the dataset for the stacking model
        if self.use_random_forests:
            self.X_stacking_train = self._get_stacking_model_dataset_rf(self.X_train)
        else:
            self.X_stacking_train = self._get_stacking_model_dataset_asl(self.X_train)
            
        # build the multi-class output for the stacking model
        best_solvers = self.orig_y_train.idxmin(axis=1)
        self.y_stacking_train = self.le_.transform(best_solvers)

        # and fit
        if self.use_random_forests:
            self.stacking_model = RandomForestClassifier(
                n_estimators=100
            )

            self.stacking_model_ = self.stacking_model.fit(
                self.X_stacking_train,
                self.y_stacking_train
            )

        else:
            self.stacking_model = automl_utils.AutoSklearnWrapper(
                estimator_named_step="classifer", args=self.args)

            self.stacking_model_ = self.stacking_model.fit(
                self.X_stacking_train,
                self.y_stacking_train,
                metric=metric
            )
        
        return self
            
    def fit(self, X_train, y_train, metric=autosklearn.metrics.f1_micro):
        
        self._fit_init(X_train, y_train)
        self._fit_regressors()
        self._fit_stacking_model(metric)

        return self
        
    def predict_proba(self, X_test):
        # add the stacking model features
        if self.use_random_forests:
            X_stacking_test = self._get_stacking_model_dataset_rf(X_test)
            y_proba_pred = self.stacking_model.predict_proba(X_stacking_test)
            return y_proba_pred

        X_stacking_test = self._get_stacking_model_dataset_asl(X_test)

        
        (weights, pipelines) = self.stacking_model_.ensemble_
        estimators = self.stacking_model_.get_estimators()

        # sometimes, the estimators drop rare classes
        # I believe this is because they do not appear in all of the internal
        # cross-validation folds.
        # anyway, make sure the output has the correct shape
        
        res_shape = (X_test.shape[0], len(self.le_.classes_))
        res = np.zeros(shape=res_shape)

        for w, p in zip(weights, pipelines):
            y_pred = w*p.predict_proba(X_stacking_test)

            e = automl_utils._get_asl_estimator(p, pipeline_step='classifier')
            p_classes = e.classes_
            res[:,p_classes] += y_pred
            
        return res
    
    def predict(self, X_test):
        
        # first, get the weighted predictions from each member of the ensemble
        y_pred = self.predict_proba(X_test)

        # now take the majority vote
        y_pred = y_pred.argmax(axis=1)
        
        return y_pred

class ASaslPipeline:
    def __init__(self, args, feature_steps=None, features=None, use_random_forests=False):
        self.args = args
        self.features = features
        self.feature_steps = feature_steps
        self.use_random_forests = use_random_forests

    def fit(self, scenario):
        """ Fit the pipeline using the ASlibScenario
        """

        if self.features is None:
            self.feature_columns_ = len(scenario.feature_data.columns)
            self.feature_columns_ = np.arange(self.feature_columns_, dtype=int)
        else:
            self.feature_columns_ = [
                scenario.feature_data.columns.get_loc(c)
                    for c in self.features
            ]

        feature_selector = mlxtend.feature_selection.ColumnSelector(
            cols=self.feature_columns_)

        nss = NaNStandardScaler()
        as_asl_ensemble = ASaslEnsemble(
            args=self.args,
            solvers=scenario.algorithms,
            use_random_forests=self.use_random_forests
        )

        # if we are using random forests, then we must also impute
        # missing values
        imputer = None
        if self.use_random_forests:
            imputer = automl_utils.get_imputer("zero_fill")
            imputer = ('imputer', imputer)
        
        pipeline = utils.remove_nones([
            ('feature_selector', feature_selector),
            ('nss', nss),
            imputer,
            ('selector', as_asl_ensemble)
        ])
        
        self.pipeline = sklearn.pipeline.Pipeline(pipeline)

        self.X_train = scenario.feature_data.values
        self.y_train = scenario.performance_data

        self.pipeline_ = self.pipeline.fit(self.X_train, self.y_train)

        return self

    def predict_proba(self, X_test):
        """ Use the fit pipeline to predict the likelihood that each solver is
        optimal for each instance
        """

        check_is_fitted(self, "pipeline_")
        y_proba_pred = self.pipeline_.predict_proba(X_test)
        return y_proba_pred

    def predict(self, X_test):
        """ Use the fit pipeline to select the solver most likely to be optimal
        for each instance
        """

        check_is_fitted(self, "pipeline_")
        y_proba_pred = self.predict_proba(X_test)
        y_pred = y_proba_pred.argmax(axis=1)
        return y_pred

    def transform(self, y, *_):
        """ Use the label encodings in the pipeline to transform the labels
        """
        check_is_fitted(self, "pipeline_")
        as_asl_ensemble = self.pipeline_.named_steps['selector']
        y_transformed = as_asl_ensemble.le_.transform(y)
        return y_transformed

    def inverse_transform(self, y, *_):
        """ Use the label encodings in the pipeline to transform the indices
        """
        check_is_fitted(self, "pipeline_")
        as_asl_ensemble = self.pipeline_.named_steps['selector']
        y_transformed = as_asl_ensemble.le_.inverse_transform(y)
        return y_transformed

            
    def create_solver_schedules(self, scenario):
        """ Use the fit pipeline to create solver schedules for the scenario
        instances
        """
        check_is_fitted(self, ["pipeline_"])

        # currently, just take the predicted best solver
        X_test = scenario.feature_data
        y_test = scenario.performance_data

        X_test = X_test.values
        y_pred = self.predict(X_test)

        msg = "X_test.shape: {}. y_pred.shape: {}".format(X_test.shape, y_pred.shape)
        logger.debug(msg)

        choices = self.inverse_transform(y_pred)
        it = zip(choices, scenario.instances)
        schedules = {}
        
        for choice, instance in it:
            if scenario.performance_type[0] == "runtime":
                solver_schedule = [[choice,scenario.algorithm_cutoff_time]]
            elif testing.performance_type[0] == "solution_quality":
                solver_schedule = [(choice,999999999999)]

            schedule = utils.remove_nones(
                self.feature_steps +
                solver_schedule
            )

            schedules[instance] = schedule

        return schedules

    def dump(self, filename):
        """ A convenience wrapper around joblib.dump
        """
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        """ A convenience wrapper around joblib.load
        """
        pipeline = joblib.load(filename)

        # and make sure we actually read the correct type of thing
        if not isinstance(pipeline, cls):
            msg = ("[as_asl_ensemble.load]: the object at {} is of type {}, "
                "but expected type was: {}".format(filename, type(pipeline),
                cls))
            raise TypeError(msg)

        return pipeline

class PresolverScheduler:
    """ Determine the presolving schedule for a dataset based
    the solver runtimes from a training set. This class constructs
    a static presolving schedule which does not depend on the instance
    features.
    
    This class uses sklearn-like nomenclature for its methods, but
    the semantics tend to be somewhat different.
    """
    def __init__(self, budget=0.1, min_fast_solutions=0.5):
        """ Create a selector
        
        Parameters
        ---------- 
        budget: float between 0 and 1
            The fraction of the scenario cutoff time used for presolving
            
        min_fast_solutions: float between 0 and 1
            The fraction of instances which must be solved "fast" to select
            a solver for presolving
        """
        self.budget = budget
        self.min_fast_solutions = min_fast_solutions
    
    def fit(self, scenario):
        """ Use the solver runtimes to construct a static presolver schedule
        
        Parameters
        ----------
        scenario: ASlibScenario
            The scenario
                    
        Returns
        -------
        self
        """
        
        # use a simple threshold to determine a "fast" solution
        self.fast_cutoff_time_ = scenario.algorithm_cutoff_time * self.budget
        self.fast_cutoff_count_ = len(scenario.instances) * self.min_fast_solutions
        
        
        self.presolver_ = None
        best_mean_fast_solution_time = np.inf

        for solver in scenario.algorithms:
            p = scenario.performance_data[solver]

            m_fast_solutions = p < self.fast_cutoff_time_
            num_fast_solutions = np.sum(m_fast_solutions)
            mean_fast_solution_time = np.sum(p[m_fast_solutions]) / num_fast_solutions
            
            # check if this solver qualifies for use as a presolver
            if num_fast_solutions > self.fast_cutoff_count_:
                if mean_fast_solution_time < best_mean_fast_solution_time:
                    best_mean_fast_solution_time = mean_fast_solution_time
                    self.presolver_ = solver

        return self

    def get_params(self, deep=False):
        params = {
            'budget': self.budget,
            'min_fast_solutions': self.min_fast_solutions
        }
        return params

    def set_params(self, **params):
        """ Set the  parameters of the presolver to the specified values
        """
        if 'budget' in params:
            self.budget = params.pop('budget')

        if 'min_fast_solutions' in params:
            self.min_fast_solutions = params.pop('min_fast_solutions')
                    
    def create_presolver_schedule(self, scenario):
        """ Create the presolver schedule for the given scenario
        """
        check_is_fitted(self, "presolver_")
        
        # check if we chose a presolver
        p = None
        if self.presolver_ is not None:
            p = [self.presolver_, self.fast_cutoff_time_]
            
        # regardless, do the same thing for all instances
        presolver_schedule = {
            instance: [p] for instance in scenario.instances
        }
        
        return presolver_schedule


class ASaslScheduler:
    """ This class chains a presolver scheduler with a feature-based scheduler
    to build complete algorithm selection schedules.

    When used in a grid search, this class assumes the feature-based scheduler
    is fixed (that is, has already been optimized using an external process).
    However, it does expose the parameters of the presolver. Thus, it is
    appropriate for optimizing the the presolver parameters wrt a metric like
    PAR10.

    Presently, the actual GridSearchCV interface from sklearn is not supported.
    """

    def __init__(self, args, config, sfss=None, feature_scheduler=None, 
            presolver_scheduler=None):
        """ Create the chained scheduler

        Parameters
        ----------
        args: argparse.ArgumentParser

        config: dict

        sfss: SequentialFeatureStepSelector

        presolver_scheduler: PresolverScheduler
            The object responsible for determining presolving schedules. The
            parameters of the presolver_scheduler can be optimizes using
            GridSearchCV-like search.

        feature_scheduler: ASaslPipeline (or similar)
            The object responsible for selecting feature sets and solver
            schedules. The parameters of the feature_scheduler are assumed
            to be fixed.
        """

        self.args = args
        self.config = config
        self.sfss = sfss
        self.presolver_scheduler = presolver_scheduler
        self.feature_scheduler = feature_scheduler

    def _fit_sfss(self, scenario):
        from as_auto_sklearn.sequential_feature_step_selector import SequentialFeatureStepSelector
        # first, select the feature steps
        sfss = SequentialFeatureStepSelector(
            self.args,
            max_feature_steps=self.args.max_feature_steps
        )

        sfss_fit = sfss.fit(scenario)

        selector_type = "rf-ensemble"
        selector_filename = filenames.get_feature_selector_filename(
            self.config['base_path'],
            selector_type,
            scenario=scenario.scenario,
            note=self.config.get('note')
        )

        sfss_fit.dump(selector_filename)
        return sfss

    def _fit_pipeline(self, scenario):
        # second, train the "main" algorithm selector using the
        # selected feature steps
        pipeline = ASaslPipeline(
            self.args,
            feature_steps=self.sfss_.get_selected_feature_steps(),
            features=self.sfss_.get_selected_features(),
            use_random_forests=self.args.use_random_forests
        )

        pipeline_fit = pipeline.fit(scenario=scenario)

        model_type = "as-asl-pipeline"
        model_filename = filenames.get_model_filename(
            self.config['base_path'],
            model_type,
            scenario=scenario.scenario,
            note=self.config.get('note')    
        )

        pipeline_fit.dump(model_filename)
        return pipeline_fit

    def _evaluate_presolver_params(self, scenario, params):
        self.presolver_scheduler.set_params(**params)
        
        all_schedules = {}
        test_scenarios = []

        for fold in self.args.folds:
            testing, training = scenario.get_split(fold)
            scheduler_fit = self.presolver_scheduler.fit(training)

            schedules_pred = self.create_schedules(testing)
            all_schedules.update(schedules_pred)

            test_scenarios.append(testing)
            
        par10 = Validator.get_par10_score(all_schedules, test_scenarios)
        return par10

    def _fit_presolver(self, scenario):
        
        presolver_grid = {
            'budget': [0, 0.001, 0.01, 0.02, 0.05, 0.1],
            'min_fast_solutions': [0.5, 0.75]
        }

        self.presolver_scheduler = PresolverScheduler()
        self.presolver_scheduler_ = self.presolver_scheduler

        best_par10 = np.inf
        best_params = None
        for params in utils.dict_product(presolver_grid):
            par10 = self._evaluate_presolver_params(scenario, params)

            if par10 < best_par10:
                best_par10 = par10
                best_params = params

        self.presolver_scheduler.set_params(**best_params)
        return self.presolver_scheduler

    def fit(self, scenario):

        self.sfss_ = self._fit_sfss(scenario)
        self.pipeline_ = self._fit_pipeline(scenario)
        self.presolver_scheduler_ = self._fit_presolver(scenario)

        return self

    def create_schedules(self, scenario):
        """ Create the algorithm selection schedules for all instances in the
        scenario
        """

        presolver_schedules = self.presolver_scheduler_.create_presolver_schedule(scenario)
        solver_schedules = self.pipeline_.create_solver_schedules(scenario)

        schedules = {}

        for instance in scenario.instances:
            
            schedule = utils.remove_nones(
                presolver_schedules[instance] +
                solver_schedules[instance]
            )

            schedules[instance] = schedule

        return schedules

    def dump(self, filename):
        """ A convenience wrapper around joblib.dump
        """
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        """ A convenience wrapper around joblib.load
        """
        pipeline = joblib.load(filename)

        # and make sure we actually read the correct type of thing
        if not isinstance(pipeline, cls):
            msg = ("[as_asl_ensemble.load]: the object at {} is of type {}, "
                "but expected type was: {}".format(filename, type(pipeline),
                cls))
            raise TypeError(msg)

        return pipeline


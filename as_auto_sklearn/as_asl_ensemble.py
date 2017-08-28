import autosklearn.metrics

import numpy as np
import pandas as pd

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.utils.validation import check_is_fitted

import misc.automl_utils as automl_utils
from misc.nan_standard_scaler import NaNStandardScaler
import misc.parallel as parallel

class ASaslEnsemble:
    def __init__(self, args, solvers):
        self.args = args
        self.solvers = solvers
        
    # train each of the regressors
    def _fit_regressor(self, solver):
        model = self.solver_asl_regressors[solver]
        y_train = self.y_train[solver]
        
        # we want to punish large errors, so use mse
        metric = autosklearn.metrics.mean_squared_error
        
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
        
    def _fit_stacking_model(self, metric=autosklearn.metrics.f1_micro):
        
        # use those to build the dataset for the stacking model
        X_stacking_train = pd.DataFrame()
        for solver in self.solvers:
            r = self.solver_asl_regressors_[solver]
            mean, std = r.predict_dist(self.X_train)
            m = "{}_mean".format(solver)
            s = "{}_std".format(solver)
            X_stacking_train[m] = mean
            X_stacking_train[s] = std

        X = [self.X_train, X_stacking_train]
        self.X_stacking_train = np.concatenate(X, axis=1)
    
        # build the multi-class output for the stacking model
        best_solvers = self.orig_y_train.idxmin(axis=1)
        self.y_stacking_train = self.le_.transform(best_solvers)

        # and fit
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
        X_stacking_test = pd.DataFrame()
        for solver in self.solvers:
            mean, std = self.solver_asl_regressors_[solver].predict_dist(X_test)
            m = "{}_mean".format(solver)
            s = "{}_std".format(solver)
            X_stacking_test[m] = mean
            X_stacking_test[s] = std

        X_stacking_test = np.concatenate([X_test, X_stacking_test], axis=1)
        
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
    def __init__(self, args):
        self.args = args

    def fit(self, scenario):
        """ Fit the pipeline using the ASlibScenario
        """
            
        nss = NaNStandardScaler()
        as_asl_ensemble = ASaslEnsemble(
            args=self.args,
            solvers=scenario.algorithms
        )
        
        pipeline = [
            ('nss', nss),
            ('selector', as_asl_ensemble)
        ]
        
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

        # currently, just take the predicted best solver
        X_test = scenario.feature_data
        y_test = scenario.performance_data

        X_test = X_test.values
        y_pred = self.predict(X_test)

        choices = self.inverse_transform(y_pred)

        it = zip(choices, scenario.instances)
        predictions = {}
        
        for choice, instance in it:
            if scenario.performance_type[0] == "runtime":
                predictions[instance] = [[choice,scenario.algorithm_cutoff_time]]
            elif testing.performance_type[0] == "solution_quality":
                predictions[instance] = [(choice,999999999999)]

        return predictions






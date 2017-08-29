import logging
logger = logging.getLogger(__name__)

import joblib
import numpy as np
import mlxtend.feature_selection
from sklearn.utils.validation import check_is_fitted

from as_auto_sklearn.as_asl_ensemble import ASaslPipeline
from as_auto_sklearn.validate import Validator

import misc.automl_utils as automl_utils

class SequentialFeatureStepSelector:
    """ This class uses a greedy, forward sequential feature search
    to select the optimal set of feature steps. Conceptually, it is
    essentially the same as the SequentialFeatureSelector from mlxtend:
    
        https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
        
    However, it includes the following differences:
    
    * It is tailored to work for the feature steps as defined for
      ASlib scenarios. Namely, it includes groups of features at
      a time, and it ensures feature dependencies are respected.
      
    * It specifically uses PAR10 as the selection criterion.
      
    * It only supports forward search.
    
    * It never places any feature steps on an "exclude" list.
    """
    def __init__(self, args, max_feature_steps=np.inf):
        self.args = args
        self.max_feature_steps = max_feature_steps
    
    def _get_par10(self, feature_steps):
        
        msg = "[SFSS]: *** evaluating feature steps: {} ***".format(feature_steps)
        logger.info(msg)
        
        total_par10 = 0.0
        total_timeouts = 0
        total_solved = 0
        total_solver_times = 0.0
        total_feature_times = 0.0
        
        total_used_presolving = 0
        
        show = (logger.getEffectiveLevel() == logging.DEBUG)
        
        for fold in self.args.folds:
            msg = "[SFSS]: evaluating fold: {}".format(fold)
            logger.info(msg)

            # first, split the scenario into training and testing
            testing, training = self.scenario.get_split(fold)
            
            msg = "[SFSS]: num testing instances: {}".format(len(testing.instances))
            logger.debug(msg)
            

            # construct and fit a pipeline with the indicated feature steps
            pipeline = ASaslPipeline(
                self.args,
                feature_steps=feature_steps,
                use_random_forests=True
            )

            pipeline_fit = pipeline.fit(scenario=training)
            
            # now, check the par10 both with and without presolving
            schedules = pipeline_fit.create_solver_schedules(testing)
            
            msg = "[SFSS]: length of schedules: {}".format(len(schedules))
            logger.debug(msg)

            # use par10 to evaluate the pipeline
            validator = Validator()
            if self.scenario.performance_type[0] == "runtime":
                stat = validator.validate_runtime(
                    schedules=schedules,
                    test_scenario=testing,
                    show=show
                )
            else:
                stat = validator.validate_quality(
                    schedules=schedules,
                    test_scenario=testing,
                    show=show
                )
                                   
            total_par10 += stat.par10
            total_timeouts += stat.timeouts
            total_solved += stat.solved
            total_solver_times += stat.solver_times
            total_feature_times += stat.feature_times
                        
        total = total_timeouts + total_solved
        total_par10 = total_par10 / total
                
        msg = [
            "",
            " feature_steps: {}".format(feature_steps), 
            " min_par10: {}".format(total_par10),
            " total_timeouts: {}".format(total_timeouts),
            " total_solved: {}".format(total_solved),
            " total_solver_times: {}".format(total_solver_times),
            " total_feature_times: {}".format(total_feature_times)
        ]
        msg = "\n".join(msg)
        logger.info(msg)

        return total_par10

    def _find_best_feature_step(self):
        """ Based on the current set of included feature steps, find
        the next best one to include
        """

        best_feature_step = None
        best_par10 = np.inf

        for feature_step in self.remaining_feature_steps_:
            test_feature_steps = self.cur_feature_steps_ + [feature_step]

            if not automl_utils.check_all_dependencies(
                    self.scenario, test_feature_steps):
                continue

            test_par10 = self._get_par10(test_feature_steps)
            if test_par10 < best_par10:
                best_par10 = test_par10
                best_feature_step = feature_step

        return (best_feature_step, best_par10)
        
    def fit(self, scenario):
        """ Select the optimal set of feature steps according to PAR10
        
        Parameters
        ----------
        scenario: ASlibScenario
            The scenario
            
        Returns
        -------
        self
        """
        self.scenario = scenario
        self.cur_feature_steps_ = []
        self.cur_par10_ = np.inf
        
        self.trajectory_ = []

        # make sure to use a copy
        self.remaining_feature_steps_ = list(scenario.feature_steps)

        while len(self.cur_feature_steps_) < self.max_feature_steps:
            (best_feature_step, best_par10) = self._find_best_feature_step()

            if best_par10 > self.cur_par10_:
                break

            self.cur_feature_steps_.append(best_feature_step)
            self.remaining_feature_steps_.remove(best_feature_step)

            self.cur_par10_ = best_par10
            
            t = (list(self.cur_feature_steps_), self.cur_par10_)
            self.trajectory_.append(t)


        self.selected_features_ = automl_utils.extract_feature_names(
            self.scenario, 
            self.cur_feature_steps_
        )

        # we cannot keep the scenario around for pickling, so forget it
        self.scenario = None
            
        return self

    def get_selected_feature_steps(self):
        """ Get the names of the selected feature steps
        """
        check_is_fitted(self, "cur_feature_steps_")
        return self.cur_feature_steps_

    def get_selected_features(self):
        """ Get the names of the selected features based on the steps
        """
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    
    def get_transformer(self):
        """ Get a ColumnSelector based on the selected feature steps
        """
              
        selected_features = self.get_selected_features()
        feature_selector = mlxtend.feature_selection.ColumnSelector(
            cols=selected_features)
        
        return feature_selector

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
            msg = ("[SFSS.load]: the object at {} is of type {}, "
                "but expected type was: {}".format(filename, type(pipeline),
                cls))
            raise TypeError(msg)

        return pipeline


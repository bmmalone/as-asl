import numpy as np
from sklearn.utils.validation import check_is_fitted

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

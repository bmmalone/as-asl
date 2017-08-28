import os

from aslib_scenario.aslib_scenario import ASlibScenario

class OascTestScenario:
    """ Read the parts of an ASlibScenario available for the test sets
    in the Open Algorithm Selection Challenge:
    
    http://www.coseal.net/open-algorithm-selection-challenge-2017-oasc/
    """
    
    def __init__(self, path):
        # read the parts of the aslib scenario which are present. This is adapted from
        # the example here: (in the predict method)
        #
        # https://github.com/mlindauer/OASC_starterkit/blob/master/oasc_starterkit/single_best.py
        
        scenario = ASlibScenario()
        scenario.read_description(fn=os.path.join(path,"description.txt"))
        scenario.read_feature_values(fn=os.path.join(path,"feature_values.arff"))
        scenario.read_feature_runstatus(fn=os.path.join(path,"feature_runstatus.arff"))
        
        self.scenario = scenario

    @classmethod
    def load_all(cls, scenarios_dir):
        """ Load all scenarios in scenarios_dir into a dictionary

        In particular, this function assumes all subdirectories within
        scenarios_dir are OASC testing scenarios

        Parameters
        ----------
        scenarios_dir: path-like
            The location of the scenarios

        Returns
        -------
        scenarios: dictionary of string -> ASlibScenario
            A dictionary where the key is the name of the scenario and the value
            is the corresponding (partial) ASlibScenario
        """
            
        # first, just grab everything in the directory
        scenarios = [
            os.path.join(scenarios_dir, o) for o in os.listdir(scenarios_dir)
        ]

        # only keep the subdirectories
        scenarios = sorted([
            t for t in scenarios if os.path.isdir(t)
        ])

        # load the scenarios
        scenarios = [
            cls(s) for s in scenarios
        ]

        # use the scenario names as keys in the dictionary
        scenarios = {
            s.scenario.scenario: s.scenario for s in scenarios
        }

        return scenarios



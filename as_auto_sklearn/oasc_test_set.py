import os

from aslib_scenario.aslib_scenario import ASlibScenario

class OascTestSet:
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

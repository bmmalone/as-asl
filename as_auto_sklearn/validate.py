import sys
import logging

from aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"
# this is modified version of the validation script from AutoFolio

class Stats(object):

    def __init__(self, runtime_cutoff):
        ''' Constructor 

            Arguments
            ---------
            runtime_cutoff: int
                maximal running time
        '''
        import collections

        self.par1 = 0.0
        self.par10 = 0.0
        self.timeouts = 0
        self.solved = 0
        self.unsolvable = 0
        self.presolved_feats = 0
        self.solver_times = 0
        self.feature_times= 0

        self.runtime_cutoff = runtime_cutoff
        
        self.selection_freq = collections.defaultdict(int)

        self.oracle_par1 = 0.0

        self.logger = logging.getLogger("Stats")

    def show(self, remove_unsolvable: bool=True):
        '''
            shows statistics

            Arguments
            --------
            remove_unsolvable : bool
                remove unsolvable from stats
                
            Returns
            -------
            par10: int
                penalized average runtime 
        '''

        if remove_unsolvable and self.runtime_cutoff:
            rm_string = "removed"
            self.logger.debug("Statistics before removing unsolvable instances")
            self.logger.debug("PAR1: %.4f" %(self.par1 / (self.timeouts + self.solved)))
            self.logger.debug("PAR10: %.4f" %(self.par10 / (self.timeouts + self.solved)))
            self.logger.debug("Timeouts: %d / %d" %(self.timeouts, self.timeouts + self.solved))
            timeouts = self.timeouts - self.unsolvable
            par1 = self.par1 - (self.unsolvable * self.runtime_cutoff)
            par10 = self.par10 - (self.unsolvable * self.runtime_cutoff * 10)
        else:
            rm_string = "not removed"
            timeouts = self.timeouts
            par1 = self.par1
            par10 = self.par10
            
        if self.runtime_cutoff:

            n_samples = timeouts + self.solved
            self.logger.info("Oracle PAR1: %.4f" % (self.oracle_par1 / n_samples))
            self.logger.info("PAR1: %.4f" % (par1 / n_samples))
            self.logger.info("PAR10: %.4f" % (par10 / n_samples))
            self.logger.info("Timeouts: %d / %d" % (timeouts, n_samples))
            self.logger.info("Presolved during feature computation: %d / %d" % (self.presolved_feats, n_samples))

            self.logger.info("Solver time: %.4f" % (self.solver_times / n_samples))
            self.logger.info("Feature time: %.4f" % (self.feature_times / n_samples))

            self.logger.info("Solved: %d / %d" % (self.solved, n_samples))
            self.logger.info("Unsolvable (%s): %d / %d" % 
                             (rm_string, self.unsolvable, n_samples+self.unsolvable))
        else:
            n_samples = self.solved
            self.logger.info("Number of instances: %d" %(n_samples))
            self.logger.info("Oracle Average Solution Quality: %.4f" % (self.oracle_par1 / n_samples))
            self.logger.info("Average Solution Quality: %.4f" % (par1 / n_samples))
            par10 = par1
            
            
        self.logger.debug("Selection Frequency")
        for algo, n in self.selection_freq.items():
            self.logger.debug("%s: %.2f" %(algo, n/(timeouts + self.solved)))
            
        return par10 / n_samples

    def get_n_samples(self):
        if self.runtime_cutoff:
            timeouts = self.timeouts - self.unsolvable
            n_samples = timeouts + self.solved
        else:
            n_samples = self.solved
        return n_samples


    def merge(self, stat):
        '''
            adds stats from another given Stats objects

            Arguments
            ---------
            stat : Stats
        '''
        self.par1 += stat.par1
        self.par10 += stat.par10
        self.timeouts += stat.timeouts
        self.solved += stat.solved
        self.unsolvable += stat.unsolvable
        self.presolved_feats += stat.presolved_feats
        
        for algo, n in stat.selection_freq.items():
            self.selection_freq[algo]  = self.selection_freq.get(algo, 0) + n

class Validator(object):

    def __init__(self, logging_level="DEBUG"):
        ''' Constructor '''
        self.logger = logging.getLogger("Validation")
        self.logger.setLevel(logging_level)

    def validate(self, schedules:dict, test_scenario:ASlibScenario, show:bool=True):
        '''
            validate selected schedules on test instances for runtime

            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
            show: bool
                whether to print the statistics after calculating them
        '''
        if test_scenario.performance_type[0] == "runtime":
            stat = self.validate_runtime(schedules, test_scenario, show)
        else:
            stat = self.validate_quality(schedules, test_scenario, show)
        return stat


    def validate_runtime(self, schedules: dict, test_scenario: ASlibScenario, show:bool=True):
        '''
            validate selected schedules on test instances for runtime

            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
            show: bool
                whether to print the statistics after calculating them
        '''
        if test_scenario.performance_type[0] != "runtime":
            raise ValueError("Cannot validate non-runtime scenario with runtime validation method")
        
        stat = Stats(runtime_cutoff=test_scenario.algorithm_cutoff_time)

        feature_times = False
        if test_scenario.feature_cost_data is not None and test_scenario.performance_type[0] == "runtime":
            feature_times = True

        ok_status = test_scenario.runstatus_data == "ok"
        unsolvable = ok_status.sum(axis=1) == 0
        stat.unsolvable += unsolvable.sum()
        
        # ensure that we got predictions for all test instances
        if set(test_scenario.instances).difference(schedules.keys()):
            self.logger.error("Missing predictions for %s" %(set(test_scenario.instances).difference(schedules.keys)))
            sys.exit(1)

        for inst in test_scenario.instances:
            schedule = schedules[inst]
            self.logger.debug("Validate: %s on %s" % (schedule, inst))

            # update the oracle behavior
            oracle_par1 = test_scenario.performance_data.loc[inst].min()
            stat.oracle_par1 += oracle_par1
            self.logger.debug("Oracle performance: {}".format(oracle_par1))

            used_time = 0            
            for entry in schedule:
                
                # entry was feature step/group
                if isinstance(entry, str):
                    if feature_times:
                        ftime = test_scenario.feature_cost_data[entry][inst]
                        self.logger.debug("Used Feature time: %f" % (ftime))
                        used_time += ftime
                        stat.feature_times += ftime
                    solved = (test_scenario.feature_runstatus_data[entry][inst] == "presolved")
                elif isinstance(entry,list): # algorithm
                    algo, budget = entry 
                    time = test_scenario.performance_data[algo][inst]
                    self.logger.debug("Alloted time %f of %s vs true time %f" %(budget, algo, time))
                    used_time += min(time, budget)
                    solved = (time <= budget) and test_scenario.runstatus_data[algo][inst] == "ok"

                    stat.solver_times += used_time

                    # also keep track of how often we select each algorithm
                    stat.selection_freq[algo] += 1

                self.logger.debug("Used time (so far): %f" %(used_time))
                
                if solved and used_time <= test_scenario.algorithm_cutoff_time:
                    stat.solved += 1
                    stat.par1 += used_time
                    self.logger.debug("Solved after %f" %(used_time))
                    break
                elif used_time >= test_scenario.algorithm_cutoff_time:
                    stat.timeouts += 1
                    stat.par1 += test_scenario.algorithm_cutoff_time
                    self.logger.debug("Timeout after %f (< %f)" % (test_scenario.algorithm_cutoff_time, used_time))
                    break

        stat.par10 = stat.par1 + 9 * \
            test_scenario.algorithm_cutoff_time * stat.timeouts
        
        if show:
            stat.show()

        return stat

    def validate_quality(self, schedules: dict, test_scenario: ASlibScenario, show:bool=True):
        '''
            validate selected schedules on test instances for solution quality

            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
            show: bool
                whether to print the statistics after calculating them
        '''
        if test_scenario.performance_type[0] != "solution_quality":
            raise ValueError("Cannot validate non-solution_quality scenario with solution_quality validation method")
        
        self.logger.debug("FYI: Feature costs and algorithm runstatus is ignored")
        
        if test_scenario.maximize[0]:
            test_scenario.performance_data *= -1
            self.logger.debug("Removing *-1 in performance data because of maximization")
        
        stat = Stats(runtime_cutoff=None)
        
        # ensure that we got predictions for all test instances
        if set(test_scenario.instances).difference(schedules.keys()):
            self.logger.error("Missing predictions for %s" %(set(test_scenario.instances).difference(schedules.keys)))
            sys.exit(1)
        
        for inst in test_scenario.instances:
            schedule = schedules[inst]

            # update the oracle behavior
            stat.oracle_par1 += test_scenario.performance_data.loc[inst].min()

            if len(schedule) > 1:
                self.logger.error("Validate does not support schedules for solution quality")
                sys.exit(9)
                
            selected_algo = schedule[0][0]
            perf = test_scenario.performance_data[selected_algo][inst]
            
            self.logger.debug("Using %s on %s with performance %f" %(selected_algo, inst, perf))
            
            stat.par1 += perf
            stat.par10 += perf
            stat.solved += 1
        
        if show:
            stat.show(remove_unsolvable=False)
        
        return stat

    @classmethod
    def get_par10_score(cls, schedules: dict, test_scenarios, show:bool=False):
        """ Calculate the PAR10 score for the schedule on the scenario

        Parameters
        ---------
        schedules: dict {instance name -> tuples [algo, bugdet]}
            algorithm schedules per instance
        test_scenario: ASlibScenario or a list of ASlibScenarios
            ASlib scenario (or a list of them) with test instances
        show: bool
            whether to print the statistics after calculating them
        """
        import misc.utils as utils

        test_scenarios = utils.wrap_in_list(test_scenarios)
        all_stat = None

        for test_scenario in test_scenarios:
            validator = cls(logging_level='INFO')
            stat = validator.validate(schedules, test_scenario, show)

            if all_stat is None:
                all_stat = stat
            else:
                all_stat.merge(stat)
        
        par10 = all_stat.par10 / all_stat.get_n_samples()
        return par10


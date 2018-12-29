# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne, Alexander Elvers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import pickle
from collections import namedtuple

from alib import solutions, util

from vnep_approx import randomized_rounding_triumvirate

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions

ReducedBaselineSolution = namedtuple("ReducedBaselineSolution",
                             "load runtime status found_solution embedding_ratio temporal_log nu_real_req original_number_requests")

logger = util.get_logger(__name__, make_file=False, propagate=True)

class BaselineResultReducer(object):

    def __init__(self):
        pass

    def reduce_baseline_solution(self, baseline_solutions_input_pickle_name, reduced_baseline_solutions_output_pickle_name=None):

        baseline_solutions_input_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, baseline_solutions_input_pickle_name)

        reduced_baseline_solutions_output_pickle_path = None
        if reduced_baseline_solutions_output_pickle_name is None:
            file_basename = os.path.basename(baseline_solutions_input_pickle_path).split(".")[0]
            reduced_baseline_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                         file_basename + "_reduced.pickle")
        else:
            reduced_baseline_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                         baseline_solutions_input_pickle_name)

        logger.info("\nWill read from ..\n\t{} \n\t\tand store reduced data into\n\t{}\n".format(baseline_solutions_input_pickle_path, reduced_baseline_solutions_output_pickle_path))

        logger.info("Reading pickle file at {}".format(baseline_solutions_input_pickle_path))
        with open(baseline_solutions_input_pickle_path, "rb") as input_file:
            solution = pickle.load(input_file)

        ssd = solution.algorithm_scenario_solution_dictionary
        for algorithm in ssd.keys():
            logger.info(".. Reducing results of algorithm {}".format(algorithm))
            for scenario_id in ssd[algorithm].keys():
                logger.info("   .. handling scenario {}".format(scenario_id))
                for exec_id in ssd[algorithm][scenario_id].keys():
                    params, scenario = solution.scenario_parameter_container.scenario_triple[scenario_id]
                    load = dict([((u, v), 0.0) for (u, v) in scenario.substrate.edges])
                    for u in scenario.substrate.nodes:
                        for types in scenario.substrate.node[u]['supported_types']:
                            load[(types, u)] = 0.0
                    mappings = ssd[algorithm][scenario_id][exec_id].solution.request_mapping
                    number_of_embedde_reqs = 0
                    number_of_req_profit = 0
                    number_of_requests = len(ssd[algorithm][scenario_id][exec_id].solution.scenario.requests)
                    for req in ssd[algorithm][scenario_id][exec_id].solution.scenario.requests:
                        if req.profit > 0.001:
                            number_of_req_profit += 1
                        if mappings[req].is_embedded:
                            number_of_embedde_reqs += 1
                            for i, u in mappings[req].mapping_nodes.iteritems():
                                node_demand = req.get_node_demand(i)
                                load[(req.get_type(i), u)] += node_demand
                            for ve, sedge_list in mappings[req].mapping_edges.iteritems():
                                edge_demand = req.get_edge_demand(ve)
                                for sedge in sedge_list:
                                    load[sedge] += edge_demand
                    percentage_embbed = number_of_embedde_reqs / float(number_of_requests)
                    algo_result = ssd[algorithm][scenario_id][exec_id]
                    ssd[algorithm][scenario_id][exec_id] = ReducedBaselineSolution(
                        load=load,
                        runtime=algo_result.temporal_log.log_entries[-1].time_within_gurobi,
                        status=algo_result.status,
                        found_solution=None,
                        embedding_ratio=percentage_embbed,
                        temporal_log=algo_result.temporal_log,
                        nu_real_req=number_of_req_profit,
                        original_number_requests=number_of_requests
                    )
        del solution.scenario_parameter_container.scenario_list
        del solution.scenario_parameter_container.scenario_triple

        logger.info("Writing result pickle to {}".format(reduced_baseline_solutions_output_pickle_path))
        with open(reduced_baseline_solutions_output_pickle_path, "wb") as f:
            pickle.dump(solution, f)
        logger.info("All done.")

class RandRoundResultReducer(object):

    def __init__(self):
        pass

    def reduce_randomized_rounding_solution(self,
                                            randround_solutions_input_pickle_name,
                                            reduced_randround_solutions_output_pickle_name=None):

        randround_solutions_input_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR,
                                                             randround_solutions_input_pickle_name)

        reduced_randround_solutions_output_pickle_path = None
        if reduced_randround_solutions_output_pickle_name is None:
            file_basename = os.path.basename(randround_solutions_input_pickle_path).split(".")[0]
            reduced_randround_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                          file_basename + "_reduced.pickle")
        else:
            reduced_randround_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                          randround_solutions_input_pickle_name)

        logger.info("\nWill read from ..\n\t{} \n\t\tand store reduced data into\n\t{}\n".format(
            randround_solutions_input_pickle_path, reduced_randround_solutions_output_pickle_path))

        logger.info("Reading pickle file at {}".format(randround_solutions_input_pickle_path))
        with open(randround_solutions_input_pickle_path, "rb") as f:
            sss = pickle.load(f)

        sss.scenario_parameter_container.scenario_list = None
        sss.scenario_parameter_container.scenario_triple = None

        for alg, scenario_solution_dict in sss.algorithm_scenario_solution_dictionary.iteritems():
            logger.info(".. Reducing results of algorithm {}".format(alg))
            for sc_id, ex_param_solution_dict in scenario_solution_dict.iteritems():
                logger.info("   .. handling scenario {}".format(sc_id))
                for ex_id, solution in ex_param_solution_dict.iteritems():
                    compressed = self.reduce_single_solution(solution)
                    ex_param_solution_dict[ex_id] = compressed

        logger.info("Writing result pickle to {}".format(reduced_randround_solutions_output_pickle_path))
        with open(os.path.join(reduced_randround_solutions_output_pickle_path),
                  "w") as f:
            pickle.dump(sss, f)
        logger.info("All done.")

    def reduce_single_solution(self, solution):
        if solution is None:
            return None
        avg_runtime = self.get_avg_runtime(solution)
        best_feasible = self.get_best_feasible_or_least_violating_solution(solution)
        best_objective = self.get_highest_obj_sol(solution)
        del solution.collection_of_samples_with_violations[:]

        # set the time of both to avg_runtime
        best_feasible = best_feasible._asdict()
        best_feasible["time_to_round_solution"] = avg_runtime
        best_feasible = randomized_rounding_triumvirate.RandomizedRoundingSolutionData(**best_feasible)

        best_objective = best_objective._asdict()
        best_objective["time_to_round_solution"] = avg_runtime
        best_objective = randomized_rounding_triumvirate.RandomizedRoundingSolutionData(**best_objective)

        solution.collection_of_samples_with_violations.append(best_feasible)
        solution.collection_of_samples_with_violations.append(best_objective)
        return solution

    def get_avg_runtime(self, full_solution):
        t = 0.0
        for sample in full_solution.collection_of_samples_with_violations:
            if sample is None:
                continue
            t += sample.time_to_round_solution

        return t / len(full_solution.collection_of_samples_with_violations)

    def get_best_feasible_or_least_violating_solution(self, full_solution):
        logger.debug("Getting the best feasible solution {}".format(full_solution))
        best_sample = None
        best_max_load = None
        best_obj = None
        for sample in full_solution.collection_of_samples_with_violations:
            if sample is None:
                continue
            sample_max_load = max(sample.max_node_load, sample.max_edge_load)
            sample_obj = sample.profit

            replace_best = False
            if best_sample is None:  # initialize with any sample
                replace_best = True
            elif best_max_load > 1.0:
                if sample_max_load < best_max_load:
                    # current best result is violating the cap. and we found one with smaller violation
                    replace_best = True
            elif best_max_load <= 1.0:
                if sample_max_load <= 1.0 and sample_obj > best_obj:
                    # current best result is feasible but we found a
                    # feasible solution with a better objective
                    replace_best = True

            if replace_best:
                output = "    Replacing:\n" \
                         "          Old:" + str(best_sample) + "\n" \
                         "          New:" + str(sample) + "\n"
                best_sample = sample
                best_max_load = sample_max_load
                best_obj = sample_obj
                logger.debug(output)
            else:
                logger.debug("      Discard:" + str(sample))
        logger.debug("Best feasible with obj {} and max load {}: {}".format(best_obj, best_max_load, best_sample))
        return best_sample

    def get_highest_obj_sol(self, full_sol):
        logger.debug("Getting the solution with the greatest objective")
        best_sample = None
        best_max_load = None
        best_obj = None
        for sample in full_sol.collection_of_samples_with_violations:
            if sample is None:
                continue
            sample_max_load = max(sample.max_node_load, sample.max_edge_load)
            sample_obj = sample.profit

            replace_best = False
            if best_sample is None:  # initialize with any sample
                replace_best = True
            elif abs(sample_obj - best_obj) < 0.0001:  # approx equal objective
                replace_best = (sample_max_load < best_max_load)  # replace if smaller load
            elif sample_obj > best_obj:  # significantly better objective
                replace_best = True

            if replace_best:
                output = "    Replacing:\n" \
                         "          Old:" + str(best_sample) + "\n" \
                         "          New:" + str(sample) + "\n"
                best_sample = sample
                best_max_load = sample_max_load
                best_obj = sample_obj
                logger.debug(output)
            else:
                logger.debug("      Discard: " + str(sample))
        logger.debug("Best objective with obj {} and max load {}: {}".format(best_obj, best_max_load, best_sample))
        return best_sample

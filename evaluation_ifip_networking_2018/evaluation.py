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

"""This is the evaluation and plotting module.

This module handles all plotting related evaluation.
"""

import os
import pickle
import sys
from collections import namedtuple
from itertools import combinations, product
from time import gmtime, strftime

import matplotlib
import matplotlib.patheffects as PathEffects
from matplotlib import font_manager
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from alib import solutions, util

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions

OUTPUT_PATH = None
OUTPUT_FILETYPE = "png"

logger = util.get_logger(__name__, make_file=False, propagate=True)


class HeatmapPlotType(object):
    Simple_MCF = 0              #a plot only for ClassicMCFResult data
    Simple_RRT = 1              #a plot only for RandomizedRoundingTriumvirate data
    Comparison_MCF_vs_RRT = 2   #a plot comparing ClassicMCFResult with RandomizedRoundingTriumvirate
    VALUE_RANGE = range(Simple_MCF, Comparison_MCF_vs_RRT+1)

"""
Collection of heatmap plot specifications. Each specification corresponds to a specific plot and describes all essential
information:
- name:                 the title of the plot
- filename:             prefix of the files to be generated
- plot_type:            A HeatmapPlotType describing which data is required as input.             
- vmin and vmax:        minimum and maximum value for the heatmap
- cmap:                 the colormap that is to be used for the heatmap
- lookup_function:      which of the values shall be plotted. the input is a tuple consisting of a baseline and a randomized rounding
                        solution. The function must return a numeric value or NaN
- metric filter:        after having applied the lookup_function (returning a numeric value or NaN) the metric_filter is 
                        applied (if given) and values not matching this function are discarded.
- rounding_function:    the function that is applied for displaying the mean values in the heatmap plots
- colorbar_ticks:       the tick values (numeric) for the heatmap plot   

"""
heatmap_specification_obj = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Objective Gap [%]",
    filename="objective_gap",
    vmin=0.0,
    vmax=16.0,
    colorbar_ticks=[x for x in range(0,17,2)],
    cmap="Blues",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: mcf_result.status.objGap * 100,
    metric_filter=lambda obj: (obj >= -0.00001)
)

heatmap_specification_runtime = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Runtime [min]",
    filename="runtime",
    vmin=0,
    vmax=180,
    colorbar_ticks=[x for x in range(0,181,20)],
    cmap="Greys",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: mcf_result.temporal_log.log_entries[-1].globaltime / 60.0,
    rounding_function=lambda x: int(round(x))
)

heatmap_specification_embedding_ratio = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Acceptance Ratio [%]",
    filename="embedding_ratio",
    vmin=0.0,
    vmax=100.0,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Greens",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: mcf_result.embedding_ratio * 100.0,
)

heatmap_specification_embedding_ratio_cleaned = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: #Embedded / #Feasible [%]       ",
    filename="cleaned_embedding_ratio",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Greens",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=(lambda mcf_result:
                     ((mcf_result.embedding_ratio * mcf_result.original_number_requests / mcf_result.nu_real_req) * 100) if mcf_result.nu_real_req > 0.5
                     else np.NaN)
)

heatmap_specification_nu_real_req = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: #Feasible Requests",
    filename="real_req",
    vmin=0,
    vmax=100,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Greens",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: mcf_result.nu_real_req,
)

heatmap_specification_average_node_load = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Avg. Node Load [%]",
    filename="avg_node_load",
    vmin=0.0,
    vmax=60,
    colorbar_ticks=[x for x in range(0,61,10)],
    cmap="Oranges",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: compute_average_node_load(mcf_result),
)

heatmap_specification_average_edge_load = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Avg. Edge Load [%]",
    filename="avg_edge_load",
    vmin=0.0,
    vmax=30,
    colorbar_ticks=[x for x in range(0,31,5)],
    cmap="Purples",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: compute_average_edge_load(mcf_result),
)

heatmap_specification_max_node_load = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Max. Node Load [%]",
    filename="max_node_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Oranges",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: compute_max_node_load(mcf_result),
)

heatmap_specification_max_edge_load = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: Max. Edge Load [%]",
    filename="max_edge_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Purples",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: compute_max_edge_load(mcf_result)
)

heatmap_specification_max_load = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: MaxLoad (Edge and Node)",
    filename="max_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Reds",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: compute_max_load(mcf_result),
)

heatmap_specification_avg_load = dict(
    name="$\\mathrm{MIP}_{\\mathrm{MCF}}$: AvgLoad (Edge and Node)",
    filename="avg_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0,101,20)],
    cmap="Reds",
    plot_type=HeatmapPlotType.Simple_MCF,
    lookup_function=lambda mcf_result: compute_avg_load(mcf_result),
)

heatmap_specification_runtime_randround_preprocessing = dict(
    name="$\\mathrm{LP}_{\\mathrm{novel}}$: Runtime Pre-Processing[s]",
    filename="randround_runtime_pre",
    vmin=0,
    vmax=60,
    colorbar_ticks=[x for x in range(0,61,10)],
    cmap="Greys",
    plot_type=HeatmapPlotType.Simple_RRT,
    lookup_function=lambda rrt_result: rrt_result.meta_data.time_preprocessing
)

heatmap_specification_runtime_randround_optimization = dict(
    name="$\\mathrm{LP}_{\\mathrm{novel}}$: Runtime LP [min]",
    filename="randround_runtime_opt",
    vmin=0,
    vmax=180,
    colorbar_ticks=[x for x in range(0,181,20)],
    cmap="Greys",
    lookup_function=lambda rrt_result: rrt_result.meta_data.time_optimization / 60.0,
    plot_type=HeatmapPlotType.Simple_RRT,
    rounding_function=lambda x: int(round(x))
)

heatmap_specification_runtime_randround_postprocessing = dict(
    name="$\\mathrm{LP}_{\\mathrm{novel}}$: Runtime Post-Processing [s]",
    filename="randround_runtime_post",
    vmin=0,
    vmax=60,
    colorbar_ticks=[x for x in range(0,61,10)],
    cmap="Greys",
    plot_type=HeatmapPlotType.Simple_RRT,
    lookup_function=lambda rrt_result: rrt_result.meta_data.time_postprocessing
)

heatmap_specification_runtime_randround_runtime = dict(
    name="$\\mathrm{LP}_{\\mathrm{novel}}$: Total Runtime [min]",
    filename="randround_runtime_total",
    vmin=0,
    vmax=10,
    colorbar_ticks=[x for x in range(0,11,2)],
    cmap="Greys",
    plot_type=HeatmapPlotType.Simple_RRT,
    lookup_function=lambda rrt_result: (rrt_result.meta_data.time_preprocessing +
                                        rrt_result.meta_data.time_optimization +
                                        rrt_result.meta_data.time_postprocessing) / 60.0,
    rounding_function=lambda x: "{0:.2f}".format(x)
)

heatmap_specification_runtime_mdk_runtime = dict(
    name="Runtime MDK [s]",
    filename="mdk_runtime_total",
    vmin=0,
    vmax=60,
    colorbar_ticks=[x for x in range(0, 61, 10)],
    cmap="Greys",
    plot_type=HeatmapPlotType.Simple_RRT,
    lookup_function=lambda rrt_result: rrt_result.mdk_meta_data.time_preprocessing +
                                       rrt_result.mdk_meta_data.time_optimization +
                                       rrt_result.mdk_meta_data.time_postprocessing,
)


heatmap_specification_comparison_baseline_rr_mdk = dict(
    name="Heuristic Rounding Performance      \n$\mathrm{Profit}({\mathrm{RR}_{\mathrm{MDK}}}) / \mathrm{Profit}({\mathrm{MIP}_{\mathrm{MCF}}})$ [%]     ",
    filename="comparison_baseline_rr_mdk",
    vmin=50.0,
    vmax=100,
    colorbar_ticks=[x for x in range(50,101,10)],
    cmap="Reds",
    plot_type=HeatmapPlotType.Comparison_MCF_vs_RRT,
    lookup_function=lambda mcf_result, rrt_result: (
        (rrt_result.mdk_result.profit / mcf_result.status.objValue) * 100 if mcf_result.status.objValue > 0.000001
        else np.NaN)
)

heatmap_specification_comparison_baseline_rr_heuristic = dict(
    name="Heuristic Rounding Performance      \n$\mathrm{Profit}({\mathrm{RR}_{\mathrm{Heuristic}}}) / \mathrm{Profit}({\mathrm{MIP}_{\mathrm{MCF}}})$ [%]     ",
    filename="comparison_baseline_rr_heuristic",
    vmin=50.0,
    vmax=100,
    colorbar_ticks=[x for x in range(50,101,10)],
    cmap="Reds",
    plot_type=HeatmapPlotType.Comparison_MCF_vs_RRT,
    lookup_function=lambda mcf_result, rrt_result: (
        (rrt_result.result_wo_violations.profit / mcf_result.status.objValue) * 100 if mcf_result.status.objValue > 0.000001
        else np.NaN)
)

heatmap_specification_comparison_baseline_rr_min_load = dict(
    name="Heuristic Rounding Performance      \n$\mathrm{Profit}({\mathrm{RR}_{\mathrm{MinLoad}}}) / \mathrm{Profit}({\mathrm{MIP}_{\mathrm{MCF}}})$ [%]     ",
    filename="comparison_baseline_rr_min_load",
    vmin=95.0,
    vmax=145.0,
    colorbar_ticks=[x for x in range(95,146,10)],
    cmap="Reds",
    plot_type=HeatmapPlotType.Comparison_MCF_vs_RRT,
    lookup_function=lambda mcf_result, rrt_result: (
        (rrt_result.collection_of_samples_with_violations[0].profit / mcf_result.status.objValue) * 100 if mcf_result.status.objValue > 0.000001
        else np.NaN),
    rounding_function=lambda x: int(round(x))
)

heatmap_specification_comparison_baseline_rr_max_profit = dict(
    name="Heuristic Rounding Performance      \n$\mathrm{Profit}({\mathrm{RR}_{\mathrm{MaxProfit}}}) / \mathrm{Profit}({\mathrm{MIP}_{\mathrm{MCF}}})$ [%]     ",
    filename="comparison_baseline_rr_max_profit",
    vmin=95.0,
    vmax=145.0,
    colorbar_ticks=[x for x in range(95,146,10)],
    cmap="Reds",
    plot_type=HeatmapPlotType.Comparison_MCF_vs_RRT,
    lookup_function=lambda mcf_result, rrt_result: (
        (rrt_result.collection_of_samples_with_violations[1].profit / mcf_result.status.objValue) * 100 if mcf_result.status.objValue > 0.000001
        else np.NaN),
    rounding_function=lambda x: int(round(x))
)

global_heatmap_specfications = [
    heatmap_specification_max_node_load,
    heatmap_specification_max_edge_load,
    heatmap_specification_obj,
    heatmap_specification_runtime,
    heatmap_specification_embedding_ratio,
    heatmap_specification_average_node_load,
    heatmap_specification_average_edge_load,
    heatmap_specification_max_load,
    heatmap_specification_avg_load,
    heatmap_specification_nu_real_req,
    heatmap_specification_embedding_ratio_cleaned,
    heatmap_specification_runtime_randround_preprocessing,
    heatmap_specification_runtime_randround_optimization,
    heatmap_specification_runtime_randround_postprocessing,
    heatmap_specification_comparison_baseline_rr_mdk,
    heatmap_specification_comparison_baseline_rr_heuristic,
    heatmap_specification_comparison_baseline_rr_min_load,
    heatmap_specification_comparison_baseline_rr_max_profit,
    heatmap_specification_runtime_randround_runtime,
    heatmap_specification_runtime_mdk_runtime,
]

heatmap_specifications_per_type = {
    plot_type_item : [heatmap_specification for heatmap_specification in global_heatmap_specfications if heatmap_specification['plot_type'] == plot_type_item]
        for plot_type_item in [HeatmapPlotType.Simple_MCF, HeatmapPlotType.Simple_RRT, HeatmapPlotType.Comparison_MCF_vs_RRT]
}

"""
Axes specifications used for the heatmap plots.
Each specification contains the following elements:
- x_axis_parameter: the parameter name on the x-axis
- y_axis_parameter: the parameter name on the y-axis
- x_axis_title:     the legend of the x-axis
- y_axis_title:     the legend of the y-axis
- foldername:       the folder to store the respective plots in
"""
heatmap_axes_specification_resources = dict(
    x_axis_parameter="node_resource_factor",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Node Resource Factor",
    y_axis_title="Edge Resource Factor",
    foldername="AXES_RESOURCES"
)

heatmap_axes_specification_requests_edge_load = dict(
    x_axis_parameter="number_of_requests",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Number of Requests",
    y_axis_title="Edge Resource Factor",
    foldername="AXES_NO_REQ_vs_EDGE_RF"
)

heatmap_axes_specification_requests_node_load = dict(
    x_axis_parameter="number_of_requests",
    y_axis_parameter="node_resource_factor",
    x_axis_title="Number of Requests",
    y_axis_title="Node Resource Factor",
    foldername="AXES_NO_REQ_vs_NODE_RF"
)

global_heatmap_axes_specifications = [heatmap_axes_specification_requests_edge_load,
                                      heatmap_axes_specification_resources,
                                      heatmap_axes_specification_requests_node_load]


def compute_average_node_load(result_summary):
    logger.warn("In the function compute_average_node_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types. Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x == "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_average_edge_load(result_summary):
    logger.warn("In the function compute_average_edge_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types. Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x != "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_max_node_load(result_summary):
    logger.warn("In the function compute_max_node_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types.  Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x == "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def compute_max_edge_load(result_summary):
    logger.warn("In the function compute_max_edge_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types. Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x != "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def compute_avg_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_max_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)



def select_scenarios_with_high_objective_gap_or_zero_requests(dc_baseline, algorithm_name,
                                                              output_respective_generation_parameters=True):
    ''' Function to select scenarios with high objective gap or no requests. This function is not used anymore but
        is left here for future usage.
    '''
    scenario_ids = dc_baseline.algorithm_scenario_solution_dictionary[algorithm_name].keys()

    result = []

    for scenario_id in scenario_ids:
        scenario_solution = dc_baseline.get_solutions_by_scenario_index(scenario_id)[algorithm_name][0]
        scenario_status = scenario_solution.status
        if scenario_status.objGap > 100:
            result.append(scenario_id)

            if output_respective_generation_parameters:
                print "Scenario {} has a very high gap, i.e. a gap of {} due to the objective bound being {} and the objective value being {}".format(
                    scenario_id,
                    scenario_status.objGap,
                    scenario_status.objBound,
                    scenario_status.objValue
                )
                print "The computation for this scenario took {} seconds.".format(scenario_solution.runtime)
                print "This scenario had the following generation parameters:"
                generation_parameters = extract_generation_parameters(
                    dc_baseline.scenario_parameter_container.scenario_parameter_dict, scenario_id
                )
                for gen_param in generation_parameters:
                    print "\t {}".format(gen_param)
        if scenario_solution.nu_real_req < 0.5:
            result.append(scenario_id)

            if output_respective_generation_parameters:
                print "Scenario {} has doesn't have any reasonable scenarios in it...{}".format(scenario_id,
                                                                                                scenario_status.objGap,
                                                                                                scenario_status.objBound,
                                                                                                scenario_status.objValue)
                print "The computation for this scenario took {} seconds.".format(scenario_solution.runtime)
                print "This scenario had the following generation parameters:"
                generation_parameters = extract_generation_parameters(
                    dc_baseline.scenario_parameter_container.scenario_parameter_dict, scenario_id
                )
                for gen_param in generation_parameters:
                    print "\t {}".format(gen_param)

    print "{} many scenarios experienced a very, very high gap or contained 0 requests".format(len(result))
    return result


def get_title_for_filter_specifications(filter_specifications):
    result = "\n".join(
        [filter_specification['parameter'] + "=" + str(filter_specification['value']) + "; " for filter_specification in
         filter_specifications])
    return result[:-2]


def extract_parameter_range(scenario_parameter_space_dict, key):
    if not isinstance(scenario_parameter_space_dict, dict):
        return None
    for generator_name, value in scenario_parameter_space_dict.iteritems():
        if generator_name == key:
            return [key], value
        if isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
            result = extract_parameter_range(value, key)
            if result is not None:
                path, values = result
                return [generator_name, 0] + path, values
        elif isinstance(value, dict):
            result = extract_parameter_range(value, key)
            if result is not None:
                path, values = result
                return [generator_name] + path, values
    return None


def extract_generation_parameters(scenario_parameter_dict, scenario_id):
    if not isinstance(scenario_parameter_dict, dict):
        return None

    results = []

    for generator_name, value in scenario_parameter_dict.iteritems():
        if isinstance(value, set) and generator_name != "all" and scenario_id in value:
            return [[generator_name]]
        if isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
            result = extract_generation_parameters(value, scenario_id)
            if result is not None:
                for atomic_result in result:
                    results.append([generator_name] + atomic_result)
        elif isinstance(value, dict):
            result = extract_generation_parameters(value, scenario_id)
            if result is not None:
                for atomic_result in result:
                    results.append([generator_name] + atomic_result)

    if results == []:
        return None
    else:
        # print "returning {}".format(results)
        return results


def lookup_scenarios_having_specific_values(scenario_parameter_space_dict, path, value):
    current_path = path[:]
    current_dict = scenario_parameter_space_dict
    while len(current_path) > 0:
        if isinstance(current_path[0], basestring):
            current_dict = current_dict[current_path[0]]
            current_path.pop(0)
        elif current_path[0] == 0:
            current_path.pop(0)
    # print current_dict
    return current_dict[value]

def lookup_scenario_parameter_room_dicts_on_path(scenario_parameter_space_dict, path):
    current_path = path[:]
    current_dict_or_list = scenario_parameter_space_dict
    dicts_on_path = []
    while len(current_path) > 0:
        dicts_on_path.append(current_dict_or_list)
        if isinstance(current_path[0], basestring):
            current_dict_or_list = current_dict_or_list[current_path[0]]
            current_path.pop(0)
        elif isinstance(current_path[0], int):
            current_dict_or_list = current_dict_or_list[int(current_path[0])]
            current_path.pop(0)
        else:
            raise RuntimeError("Could not lookup dicts.")
    return dicts_on_path

def load_reduced_pickle(reduced_pickle):
    with open(reduced_pickle, "rb") as f:
        data = pickle.load(f)
    return data

class AbstractPlotter(object):
    ''' Abstract Plotter interface providing functionality used by the majority of plotting classes of this module.
    '''

    def __init__(self,
                 output_path,
                 output_filetype,
                 scenario_solution_storage,
                 algorithm_id,
                 execution_id,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        self.output_path = output_path
        self.output_filetype = output_filetype
        self.scenario_solution_storage = scenario_solution_storage

        self.algorithm_id = algorithm_id
        self.execution_id = execution_id

        self.scenario_parameter_dict = self.scenario_solution_storage.scenario_parameter_container.scenario_parameter_dict
        self.scenarioparameter_room = self.scenario_solution_storage.scenario_parameter_container.scenarioparameter_room
        self.all_scenario_ids = set(scenario_solution_storage.algorithm_scenario_solution_dictionary[self.algorithm_id].keys())

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.overwrite_existing_files = overwrite_existing_files
        if not forbidden_scenario_ids:
            self.forbidden_scenario_ids = set()
        else:
            self.forbidden_scenario_ids = forbidden_scenario_ids
        self.paper_mode=paper_mode



    def _construct_output_path_and_filename(self, title, filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(OUTPUT_FILETYPE)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        output_path = os.path.join(base, date, OUTPUT_FILETYPE, "general_plots", filter_spec_path)
        filename = os.path.join(output_path, title + "_" + filter_filename)
        return output_path, filename


    def _construct_path_and_filename_for_filter_spec(self, filter_specifications):
        filter_path = ""
        filter_filename = ""
        for spec in filter_specifications:
            filter_path = os.path.join(filter_path, (spec['parameter'] + "_" + str(spec['value'])))
            filter_filename += spec['parameter'] + "_" + str(spec['value']) + "_"
        filter_filename = filter_filename[:-1] + "." + OUTPUT_FILETYPE
        return filter_path, filter_filename


    def _obtain_scenarios_based_on_filters(self, filter_specifications=None):
        allowed_scenario_ids = set(self.all_scenario_ids)
        sps = self.scenarioparameter_room
        spd = self.scenario_parameter_dict
        if filter_specifications:
            for filter_specification in filter_specifications:
                filter_path, _ = extract_parameter_range(sps, filter_specification['parameter'])
                filter_indices = lookup_scenarios_having_specific_values(spd, filter_path,
                                                                         filter_specification['value'])
                allowed_scenario_ids = allowed_scenario_ids & filter_indices

        return allowed_scenario_ids


    def _obtain_scenarios_based_on_axis(self, axis_path, axis_value):
        spd = self.scenario_parameter_dict
        return lookup_scenarios_having_specific_values(spd, axis_path, axis_value)

    def _show_and_or_save_plots(self, output_path, filename):
        plt.tight_layout()
        if self.save_plot:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print "saving plot: {}".format(filename)
            plt.savefig(filename)
        if self.show_plot:
            plt.show()

        plt.close()


    def plot_figure(self, filter_specifications):
        raise RuntimeError("This is an abstract method")


class SingleHeatmapPlotter(AbstractPlotter):

    def __init__(self,
                 output_path,
                 output_filetype,
                 scenario_solution_storage,
                 algorithm_id,
                 execution_id,
                 heatmap_plot_type,
                 list_of_axes_specifications = global_heatmap_axes_specifications,
                 list_of_metric_specifications = None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        super(SingleHeatmapPlotter, self).__init__(output_path, output_filetype, scenario_solution_storage,
                                                   algorithm_id, execution_id, show_plot, save_plot,
                                                   overwrite_existing_files, forbidden_scenario_ids, paper_mode)
        if heatmap_plot_type is None or heatmap_plot_type not in HeatmapPlotType.VALUE_RANGE:
            raise RuntimeError("heatmap_plot_type {} is not a valid input. Must be of type HeatmapPlotType.".format(heatmap_plot_type))
        self.heatmap_plot_type = heatmap_plot_type

        if not list_of_axes_specifications:
            raise RuntimeError("Axes need to be provided.")
        self.list_of_axes_specifications = list_of_axes_specifications

        if not list_of_metric_specifications:
            self.list_of_metric_specifications = heatmap_specifications_per_type[self.heatmap_plot_type]
        else:
            for metric_specification in list_of_metric_specifications:
                if metric_specification.plot_type != self.heatmap_plot_type:
                    raise RuntimeError("The metric specification {} does not agree with the plot type {}.".format(metric_specification, self.heatmap_plot_type))
            self.list_of_metric_specifications = list_of_metric_specifications

    def _construct_output_path_and_filename(self, metric_specification, heatmap_axes_specification, filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(OUTPUT_FILETYPE)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        axes_foldername = heatmap_axes_specification['foldername']
        output_path = os.path.join(base, date, OUTPUT_FILETYPE, axes_foldername, filter_spec_path)
        filename = os.path.join(output_path, metric_specification['filename'] + "_" + filter_filename)
        return output_path, filename


    def plot_figure(self, filter_specifications):
        for axes_specification in self.list_of_axes_specifications:
            for metric_specfication in self.list_of_metric_specifications:
                self.plot_single_heatmap_general(metric_specfication, axes_specification, filter_specifications)


    def _lookup_solutions(self, scenario_ids):
        return [(self.scenario_solution_storage.get_solutions_by_scenario_index(x)[self.algorithm_id][self.execution_id],) for x in scenario_ids]

    def plot_single_heatmap_general(self,
                                    heatmap_metric_specification,
                                    heatmap_axes_specification,
                                    filter_specifications=None):
        # data extraction

        sps = self.scenarioparameter_room
        spd = self.scenario_parameter_dict

        output_path, filename = self._construct_output_path_and_filename(heatmap_metric_specification,
                                                                         heatmap_axes_specification,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        #check if filter specification conflicts with axes specification
        if filter_specifications is not None:
            for filter_specification in filter_specifications:
                if (heatmap_axes_specification['x_axis_parameter'] == filter_specification['parameter'] or
                        heatmap_axes_specification['y_axis_parameter'] == filter_specification['parameter']):
                    logger.debug("Skipping generation of {} as the filter specification conflicts with the axes specification.")
                    return

        path_x_axis, xaxis_parameters = extract_parameter_range(sps, heatmap_axes_specification['x_axis_parameter'])
        path_y_axis, yaxis_parameters = extract_parameter_range(sps, heatmap_axes_specification['y_axis_parameter'])

        # for heatmap plot
        xaxis_parameters.sort()
        yaxis_parameters.sort()

        # all heatmap values will be stored in X
        X = np.zeros((len(yaxis_parameters), len(xaxis_parameters)))
        column_labels = yaxis_parameters
        row_labels = xaxis_parameters
        fig, ax = plt.subplots(figsize=(5, 4))

        min_number_of_observed_values = 10000000000000
        max_number_of_observed_values = 0
        observed_values = np.empty(0)

        for x_index, x_val in enumerate(xaxis_parameters):
            # all scenario indices which has x_val as xaxis parameter (e.g. node_resource_factor = 0.5
            scenario_ids_matching_x_axis = lookup_scenarios_having_specific_values(spd, path_x_axis, x_val)
            for y_index, y_val in enumerate(yaxis_parameters):
                scenario_ids_matching_y_axis = lookup_scenarios_having_specific_values(spd, path_y_axis, y_val)

                filter_indices = self._obtain_scenarios_based_on_filters(filter_specifications)
                scenario_ids_to_consider = (scenario_ids_matching_x_axis &
                                scenario_ids_matching_y_axis &
                                filter_indices) - self.forbidden_scenario_ids


                solutions = self._lookup_solutions(scenario_ids_to_consider)

                values = [heatmap_metric_specification['lookup_function'](*solution) for solution in solutions]

                if 'metric_filter' in heatmap_metric_specification:
                    values = [value for value in values if heatmap_metric_specification['metric_filter'](value)]

                observed_values = np.append(observed_values, values)

                if len(values) < min_number_of_observed_values:
                    min_number_of_observed_values = len(values)
                if len(values) > max_number_of_observed_values:
                    max_number_of_observed_values = len(values)

                logger.debug("values are {}".format(values))
                m = np.nanmean(values)
                logger.debug("mean is {}".format(m))

                if 'rounding_function' in heatmap_metric_specification:
                    rounded_m = heatmap_metric_specification['rounding_function'](m)
                else:
                    rounded_m = float("{0:.1f}".format(round(m, 2)))

                plt.text(x_index + .5,
                         y_index + .45,
                         rounded_m,
                         verticalalignment="center",
                         horizontalalignment="center",
                         fontsize=17.5,
                         fontname="Courier New",
                         # family="monospace",
                         color='w',
                         path_effects=[PathEffects.withStroke(linewidth=4, foreground="k")]
                         )

                X[y_index, x_index] = rounded_m

        if min_number_of_observed_values == max_number_of_observed_values:
            solution_count_string = "{} values per square".format(min_number_of_observed_values)
        else:
            solution_count_string = "between {} and {} values per square".format(min_number_of_observed_values,
                                                                                 max_number_of_observed_values)

        if self.paper_mode:
            ax.set_title(heatmap_metric_specification['name'], fontsize=17)
        else:
            title = heatmap_metric_specification['name'] + "\n"
            if filter_specifications:
                title += get_title_for_filter_specifications(filter_specifications) + "\n"
            title += solution_count_string + "\n"
            title += "min: {:.2f}; mean: {:.2f}; max: {:.2f}".format(np.nanmin(observed_values),
                                                         np.nanmean(observed_values),
                                                         np.nanmax(observed_values))

            ax.set_title(title)

        heatmap = ax.pcolor(X,
                            cmap=heatmap_metric_specification['cmap'],
                            vmin=heatmap_metric_specification['vmin'],
                            vmax=heatmap_metric_specification['vmax'])

        if not self.paper_mode:
            fig.colorbar(heatmap, label=heatmap_metric_specification['name'] + ' - mean in blue')
        else:
            ticks = heatmap_metric_specification['colorbar_ticks']
            tick_labels = [str(tick).ljust(3) for tick in ticks]
            cbar = fig.colorbar(heatmap)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
            #for label in cbar.ax.get_yticklabels():
            #    label.set_fontproperties(font_manager.FontProperties(family="Courier New",weight='bold'))

            cbar.ax.tick_params(labelsize=15.5)

        ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)

        ax.set_xticklabels(row_labels, minor=False, fontsize=15.5)
        ax.set_xlabel(heatmap_axes_specification['x_axis_title'], fontsize=16)
        ax.set_ylabel(heatmap_axes_specification['y_axis_title'], fontsize=16)
        ax.set_yticklabels(column_labels, minor=False, fontsize=15.5)

        self._show_and_or_save_plots(output_path, filename)


class ComparisonHeatmapPlotter(SingleHeatmapPlotter):

    def __init__(self,
                 output_path,
                 output_filetype,
                 scenario_solution_storage,
                 algorithm_id,
                 execution_id,
                 other_scenario_solution_storage,
                 other_algorithm_id,
                 other_execution_id,
                 heatmap_plot_type,
                 list_of_axes_specifications = global_heatmap_axes_specifications,
                 list_of_metric_specifications = None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        super(ComparisonHeatmapPlotter, self).__init__(output_path,
                                                       output_filetype,
                                                       scenario_solution_storage,
                                                       algorithm_id,
                                                       execution_id,
                                                       heatmap_plot_type,
                                                       list_of_axes_specifications,
                                                       list_of_metric_specifications,
                                                       show_plot,
                                                       save_plot,
                                                       overwrite_existing_files,
                                                       forbidden_scenario_ids,
                                                       paper_mode)
        self.other_scenario_solution_storage = other_scenario_solution_storage
        self.other_algorithm_id = other_algorithm_id
        self.other_execution_id = other_execution_id

        if heatmap_plot_type != HeatmapPlotType.Comparison_MCF_vs_RRT:
            raise RuntimeError("Only comparison heatmap plots are allowed")

    def _lookup_solutions(self, scenario_ids):
        return [(self.scenario_solution_storage.get_solutions_by_scenario_index(x)[self.algorithm_id][self.execution_id],
                 self.other_scenario_solution_storage.get_solutions_by_scenario_index(x)[self.other_algorithm_id][self.other_execution_id])
                for x in scenario_ids]




class ComparisonBaselineVsRRT_Scatter_and_ECDF(AbstractPlotter):

    def __init__(self,
                 output_path,
                 output_filetype,
                 baseline_solution_storage,
                 baseline_algorithm_id,
                 baseline_execution_id,
                 randround_solution_storage,
                 randround_algorithm_id,
                 randround_execution_id,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        super(ComparisonBaselineVsRRT_Scatter_and_ECDF, self).__init__(output_path, output_filetype, baseline_solution_storage,
                                                                       baseline_algorithm_id, baseline_execution_id, show_plot, save_plot,
                                                                       overwrite_existing_files, forbidden_scenario_ids, paper_mode)
        if randround_algorithm_id != "RandomizedRoundingTriumvirate":
            raise RuntimeError("The capacity violation plot can only be applied to RandomizedRoundingTriumvirate results.")

        self.randround_solution_storage = randround_solution_storage
        self.randround_algorithm_id = randround_algorithm_id
        self.randround_execution_id = randround_execution_id

        self._randround_data_names = ['min_aug', 'max_profit', 'wo_viol', 'mdk']
        self._randround_data_names_with_baseline = ['min_aug', 'max_profit', 'wo_viol', 'mdk', "baseline"]

        self.label_names = {'min_aug':      "min. augmentation",
                            'max_profit':   "max. profit",
                            'wo_viol':      "rounding w/o augmentation",
                            'mdk':          "multi-dimensional knapsack",
                            'baseline':     "baseline"}


        self.math_label_names = {'min_aug':      "\mathrm{RR}_{\mathrm{MinLoad}}",
                                 'max_profit':   "\mathrm{RR}_{\mathrm{MaxProfit}}",
                                 'wo_viol':      "\mathrm{RR}_{\mathrm{Heuristic}}",
                                 'mdk':          "\mathrm{RR}_{\mathrm{MDK}}",
                                 'baseline':     "\mathrm{MIP}_{\mathrm{MCF}}"}

        self.markers = {'min_aug':      "o",
                        'max_profit':   "v",
                        'wo_viol':      "x",
                        'mdk':          "+",
                        'baseline':     "^"}

        self.colors = {'min_aug':       "salmon",
                        'max_profit':   "darkred",
                        'wo_viol':      "g",
                        'mdk':          "b",
                        'baseline':     "k"}

        self._randround_data_lookups = {'min_aug': (lambda x: x.collection_of_samples_with_violations[0]),
                                        'max_profit': (lambda x: x.collection_of_samples_with_violations[1]),
                                        'wo_viol': (lambda x: x.result_wo_violations),
                                        'mdk': (lambda x: x.mdk_result)}

        filter_path_number_of_requests, list_number_of_requests = extract_parameter_range(self.scenarioparameter_room, "number_of_requests")

        self._number_of_requests_list = list_number_of_requests
        self._filter_path_number_of_requests = filter_path_number_of_requests

        self._nan_dict = {randround_data_name : np.NaN for randround_data_name in self._randround_data_names}

        self._profit_result_data_list = {randround_data_name: np.NaN for randround_data_name in
                                         self._randround_data_names}

        self._profit_result = self._nan_dict
        self._load_result = {randround_data_name: [np.NaN, np.NaN] for randround_data_name in self._randround_data_names}


    def _lookup_baseline_solution(self, scenario_id):
        return self.scenario_solution_storage.get_solutions_by_scenario_index(scenario_id)[self.algorithm_id][self.execution_id]

    def _lookup_randround_solution(self, scenario_id):
        return self.randround_solution_storage.get_solutions_by_scenario_index(scenario_id)[self.randround_algorithm_id][self.randround_execution_id]

    def _compute_profits_relative_to_baseline(self, baseline_solution, randround_solution):
        baseline_objective = baseline_solution.status.objValue
        if baseline_objective > 0.00001:
            self._profit_result = self._profit_result_data_list
            for randround_data_name in self._randround_data_names:
                randround_solution_for_data_name = self._randround_data_lookups[randround_data_name](randround_solution)
                self._profit_result[randround_data_name] = (randround_solution_for_data_name.profit / baseline_objective)*100.0
        else:
            logger.warn(
                "The baseline objective of is zero. discarding value.")
            self._profit_result = self._nan_dict

    def _compute_maximal_load_for_randround(self, randround_solution):
        for randround_data_name in self._randround_data_names:
            randround_solution_for_data_name = self._randround_data_lookups[randround_data_name](randround_solution)
            self._load_result[randround_data_name][0] = randround_solution_for_data_name.max_node_load * 100.0
            self._load_result[randround_data_name][1] = randround_solution_for_data_name.max_edge_load * 100.0


    def _extract_first_dual_bound_from_baseline_solution(self, baseline_solution):
        log_time_root = 100000000000
        root_entry = baseline_solution.temporal_log.root_relaxation_entry

        root_entry_dual_bound = -(10 ** 80)
        first_log_entry_dual_bound = -(10 ** 80)
        if root_entry is not None:
            root_entry_dual_bound = root_entry.data.objective_bound
        else:
            logger.debug("The root entry is none...")

        first_log_entry = baseline_solution.temporal_log.log_entries[0]

        if first_log_entry is not None:
            first_log_entry_dual_bound = first_log_entry.data.objective_bound
        else:
            logger.debug("The first entry of the temporal log is none...")

        result = max(root_entry_dual_bound, first_log_entry_dual_bound)
        if result < -(10 **40):
            logger.warn("The dual bound of the MIP is garbage. discarding it.")
            return np.nan
        else:
            return result

    def _extract_final_dual_bound_from_baseline_solution(self, baseline_solution):
        best_bnd = (10 **80)
        for log_entry in baseline_solution.temporal_log.log_entries:
            if log_entry.data.objective_bound < best_bnd:
                best_bnd = log_entry.data.objective_bound

        if best_bnd > 10**70:
            logger.warn("Best bound of MIP could not be determined.")
            return np.NaN
        else:
            return best_bnd


    def _compute_relative_dual_bound_to_randround_ROOT(self, baseline_solution, randround_solution):
        baseline_dual_bound = self._extract_first_dual_bound_from_baseline_solution(baseline_solution)
        randround_dual_bound = randround_solution.meta_data.status.objValue

        if randround_dual_bound > 0.0001:
            result = baseline_dual_bound / randround_dual_bound
            if result > 1000:
                logger.warn("The relative dual bound {} is very high. It's a result from {} {}. discarding it.".format(result, baseline_dual_bound, randround_dual_bound))
                return np.nan
            return result
        else:
            logger.warn(
                "The randround dual bound is zero. discarding value.")
            return np.NaN


    def _compute_relative_dual_bound_to_randround_FINAL(self, baseline_solution, randround_solution):
        baseline_dual_bound = self._extract_final_dual_bound_from_baseline_solution(baseline_solution)
        randround_dual_bound = randround_solution.meta_data.status.objValue

        if randround_dual_bound > 0.0001:
            result = baseline_dual_bound / randround_dual_bound
            if result > 1000:
                logger.warn("The relative dual bound {} is very high. It's a result from {} {}. discarding it.".format(result, baseline_dual_bound, randround_dual_bound))
                return np.nan
            return result
        else:
            logger.warn(
                "The randround dual bound is zero. discarding value.")
            return np.NaN

    def compute_relative_profits_arrays(self, list_of_scenarios):
        number_of_entries = len(list_of_scenarios)
        result = {randround_data_name: np.full(number_of_entries, np.nan) for randround_data_name in
                  self._randround_data_names}
        for i, scenario_id in enumerate(list_of_scenarios):
            baseline_solution = self._lookup_baseline_solution(scenario_id)
            randround_solution = self._lookup_randround_solution(scenario_id)
            self._compute_profits_relative_to_baseline(baseline_solution, randround_solution)
            for randround_data_name in self._randround_data_names:
                result[randround_data_name][i] = self._profit_result[randround_data_name]

        return result


    def compute_maximal_load_arrays(self, list_of_scenarios):
        number_of_entries = len(list_of_scenarios)
        result = {data_name: [np.full(number_of_entries, np.nan), np.full(number_of_entries, np.nan)]
                    for data_name in self._randround_data_names_with_baseline}
        for i, scenario_id in enumerate(list_of_scenarios):
            baseline_solution = self._lookup_baseline_solution(scenario_id)
            randround_solution = self._lookup_randround_solution(scenario_id)

            self._compute_maximal_load_for_randround(randround_solution)
            for randround_data_name in self._randround_data_names:
                result[randround_data_name][0][i] = self._load_result[randround_data_name][0]
                result[randround_data_name][1][i] = self._load_result[randround_data_name][1]

            result['baseline'][0][i] = compute_max_node_load(baseline_solution)
            result['baseline'][1][i] = compute_max_edge_load(baseline_solution)

        return result


    def compute_dual_bound_array(self, list_of_scenarios):
        result = {number_of_requests: None for number_of_requests in
                  self._number_of_requests_list}

        for number_of_requests in self._number_of_requests_list:
            scenario_ids_with_right_number_of_requests = self._obtain_scenarios_based_on_filters([{"parameter": "number_of_requests", "value": number_of_requests}])
            scenario_ids_with_right_number_of_requests &= set(list_of_scenarios)
            result[number_of_requests] = [np.full(len(scenario_ids_with_right_number_of_requests), np.nan), np.full(len(scenario_ids_with_right_number_of_requests), np.nan)]
            for i, scenario_id in enumerate(scenario_ids_with_right_number_of_requests):
                baseline_solution = self._lookup_baseline_solution(scenario_id)
                randround_solution = self._lookup_randround_solution(scenario_id)
                result[number_of_requests][0][i] = self._compute_relative_dual_bound_to_randround_ROOT(baseline_solution, randround_solution)
                result[number_of_requests][1][i] = self._compute_relative_dual_bound_to_randround_FINAL(baseline_solution, randround_solution)

        return result


    def plot_figure(self, filter_specifications):
        self.plot_figure_ecdf_load(filter_specifications)
        self.plot_figure_ecdf_objective(filter_specifications)
        self.plot_bound_ecdf(filter_specifications)
        self.plot_scatter_obj_vs_load(filter_specifications)

    def plot_figure_ecdf_load(self, filter_specifications):

        output_filename = "ECDF_load"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return


        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        result = self.compute_maximal_load_arrays(scenario_ids)

        fix, ax = plt.subplots(figsize=(5, 4))

        #cum_line = matplotlib.lines.Line2D([], [], color='k', linestyle="-", label='total')
        node_line = matplotlib.lines.Line2D([], [], color='gray', linestyle="-.", label='node')
        edge_line = matplotlib.lines.Line2D([], [], color='gray', linestyle="-", label='edge')

        second_legend_handlers = []
        max_observed_value = 0
        for data_name in self._randround_data_names_with_baseline:
            #sorted_data_cum = np.sort(np.maximum(result[data_name][0], result[data_name][1]))
            sorted_data_node = np.sort(result[data_name][0])
            sorted_data_edge = np.sort(result[data_name][1])
            max_observed_value = np.maximum(max_observed_value, sorted_data_node[-1])
            max_observed_value = np.maximum(max_observed_value, sorted_data_edge[-1])

            yvals = np.arange(1,len(sorted_data_node)+1) / float(len(sorted_data_node))

            second_legend_handlers.append(matplotlib.lines.Line2D([], [], color=self.colors[data_name], linestyle="-", label="${}$".format(self.math_label_names[data_name])))

            #ax.plot(sorted_data_cum, yvals, color=self.colors[data_name], linestyle="-")
            ax.plot(sorted_data_node, yvals, color=self.colors[data_name], linestyle="-.")
            ax.plot(sorted_data_edge, yvals, color=self.colors[data_name], linestyle="-")

        first_legend = plt.legend(handles=[node_line, edge_line], loc=4, fontsize=14, title="Resource", handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(first_legend.get_title(), fontsize=14)
        plt.gca().add_artist(first_legend)
        second_legend = plt.legend(handles=second_legend_handlers, loc=2, fontsize=14, title="Algorithm", handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(second_legend.get_title(), fontsize=14)

        ax.set_xlim(10, max_observed_value * 1.1)
        ax.set_xscale("log", basex=10)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax.set_xticks([10, 50, 100, 200, 500], minor=False)
        ax.set_xticks([20,30,40,50,60,70,80,90, 300,400], minor=True)

        ax.set_title("ECDF of Resource Loads",fontsize=17)
        ax.set_xlabel("Maximum Resource Load [%]", fontsize=16)
        ax.set_ylabel("ECDF", fontsize=16)
        ax.grid(True, which="both")


        ax.tick_params(axis='both', which='major', labelsize=15.5)
        ax.tick_params(axis='x', which='minor', labelsize=15.5)
        plt.grid(True, which="both")

        plt.tight_layout()

        self._show_and_or_save_plots(output_path, filename)



    def plot_figure_ecdf_objective(self, filter_specifications):

        output_filename = "ECDF_objective"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return


        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        result = self.compute_relative_profits_arrays(scenario_ids)

        fix, ax = plt.subplots(figsize=(5, 4))

        max_observed_value = 0
        for data_name in self._randround_data_names:
            sorted_data = np.sort(result[data_name])
            max_observed_value = np.maximum(max_observed_value, sorted_data[-1])

            yvals = np.arange(1,len(sorted_data)+1) / float(len(sorted_data))

            ax.plot(sorted_data, yvals, color=self.colors[data_name], linestyle="-", label="${}$".format(self.math_label_names[data_name]))

        leg = plt.legend(loc=4, title="Algorithm", fontsize=14, handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(leg.get_title(), fontsize=14)

        ax.set_title("ECDF of Relative Achieved Profit", fontsize=17)
        ax.set_xlabel("$\mathrm{Profit}({\mathrm{RR}_{\mathrm{Alg}}}) / \mathrm{Profit}({\mathrm{MIP}_{\mathrm{MCF}}})$ [%] ", fontsize=16)
        ax.set_ylabel("ECDF", fontsize=16)
        ax.grid(True, which="both")
        ax.tick_params(axis='both', which='major', labelsize=15.5)
        #ax.set_xscale("log", basex=10)
        ax.set_xlim(20,max_observed_value*1.1)
        plt.tight_layout()

        self._show_and_or_save_plots(output_path, filename)


    def plot_bound_ecdf(self, filter_specifications):

        output_filename = "ECDF_bound"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        if filter_specifications:
            for filter_specification in filter_specifications:
                if filter_specification["parameter"] == "number_of_requests":
                    logger.info("Skipping generation of {} as this conflicts with the filter specification {}".format(output_filename, filter_specification))
                    return

        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        result = self.compute_dual_bound_array(scenario_ids)

        fix, ax = plt.subplots(figsize=(5, 4))
        #ax.set_xscale("log", basex=10)

        colors = ['k','g', 'b', 'r']
        max_observed_value = 0

        number_requests_legend_handlers = []

        for i, number_of_requests in enumerate(self._number_of_requests_list):

            result_for_requests = result[number_of_requests][0]
            sorted_data = np.sort(result_for_requests[~np.isnan(result_for_requests)])
            max_observed_value = np.maximum(max_observed_value, sorted_data[-1])
            yvals = np.arange(1,len(sorted_data)+1) / float(len(sorted_data))
            ax.plot(sorted_data, yvals, color=colors[i], linestyle="-", label="{}".format(number_of_requests), linewidth=1.8)

            result_for_requests = result[number_of_requests][1]
            sorted_data = np.sort(result_for_requests[~np.isnan(result_for_requests)])
            max_observed_value = np.maximum(max_observed_value, sorted_data[-1])
            yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
            ax.plot(sorted_data, yvals, color=colors[i], linestyle=":",
                    linewidth=2.4)

            number_requests_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors[i], linestyle="-", label='{}'.format(number_of_requests)))

        root_legend_handlers = [matplotlib.lines.Line2D([], [], color='gray', linestyle="-", label='initial'), matplotlib.lines.Line2D([], [], color='gray', linestyle=":", label='final')]

        first_legend = plt.legend(title="Bound($\mathrm{MIP}_{\mathrm{MCF}})$", handles=root_legend_handlers, loc=(0.225,0.0125), fontsize=14, handletextpad=0.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(first_legend.get_title(), fontsize='15')
        plt.gca().add_artist(first_legend)
        o_leg = plt.legend(handles=number_requests_legend_handlers, loc=4, title="#Requests", fontsize=14, handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(o_leg.get_title(), fontsize='15')

        ax.set_title("$\mathrm{LP}_{\mathrm{novel}}$: Formulation Strength", fontsize=17)
        ax.set_xlabel("Bound($\mathrm{MIP}_{\mathrm{MCF}}$) / Bound($\mathrm{LP}_{\mathrm{novel}}$)", fontsize=16)
        ax.set_ylabel("ECDF", fontsize=16)

        ax.set_xlim(0.65,max_observed_value*1.05)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15.5)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15.5)

        ax.set_xticks([ 1, 1.5, 2, 2.5, 3, 3.5], minor=False)
        ax.set_xticks([0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25, 3.5], minor=True)
        ax.set_yticks([x*0.1 for x in range(1,10)], minor=True)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax.set_xticklabels([], minor=True)

        ax.grid(True, which="both", linestyle=":")

        # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        # for line in gridlines:
        #     line.set_linestyle(':')



        self._show_and_or_save_plots(output_path, filename)


    def plot_scatter_obj_vs_load(self, filter_specifications):

        bounding_boxes = {'min_aug':      [[50,140],[85,235]],
                          'max_profit':   [[95,210],[90,505]],
                          'wo_viol':      [[30  ,105],[75,102]],
                          'mdk':          [[15,105],[75,102]]}


        for data_to_plot in self._randround_data_names:

            bounding_box_x = bounding_boxes[data_to_plot][0]
            bounding_box_y = bounding_boxes[data_to_plot][1]

            output_filename = "SCATTER_obj_vs_load_{}".format(data_to_plot)

            output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                             filter_specifications)

            logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

            if not self.overwrite_existing_files and os.path.exists(filename):
                logger.info("Skipping generation of {} as this file already exists".format(filename))
                return

            fix, ax = plt.subplots(figsize=(5, 4))
            colors = list(self.colors.values())
            markers = list(self.markers.values())

            filter_path_NRF, node_resource_factors = extract_parameter_range(self.scenarioparameter_room,
                                                                             "node_resource_factor")

            filter_path_ERF, edge_resource_factors = extract_parameter_range(self.scenarioparameter_room,
                                                                             "edge_resource_factor")

            color_norm = matplotlib.colors.Normalize(vmin=0, vmax=6)
            scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap='inferno')

            observed_values_relative_profit = np.empty(0)
            observed_values_load = np.empty(0)

            number_of_not_shown_values = 0

            for i, nrf in enumerate(node_resource_factors):
                for j, erf in enumerate(edge_resource_factors):

                    scenario_ids = self._obtain_scenarios_based_on_filters([] +
                                                                           [
                                                                               {"parameter": "node_resource_factor",
                                                                                "value": nrf},
                                                                               {"parameter": "edge_resource_factor",
                                                                                "value": erf},
                                                                           ])
                    if self.forbidden_scenario_ids:
                        scenario_ids = scenario_ids - self.forbidden_scenario_ids

                    list_of_scenarios = list(scenario_ids)

                    result_relative_profits = self.compute_relative_profits_arrays(list_of_scenarios)

                    raw_result_loads = self.compute_maximal_load_arrays(list_of_scenarios)

                    result_cum_loads = np.maximum(raw_result_loads[data_to_plot][0],
                                                  raw_result_loads[data_to_plot][1])


                    observed_values_load = np.append(observed_values_load, result_cum_loads)

                    observed_values_relative_profit = np.append(observed_values_relative_profit,
                                                                result_relative_profits[data_to_plot])

                    for y_value in np.nditer(result_cum_loads):
                        if y_value < bounding_box_y[0] or y_value > bounding_box_y[1]:
                            number_of_not_shown_values += 1

                    for x_value in np.nditer(result_relative_profits[data_to_plot]):
                        if x_value < bounding_box_x[0] or x_value > bounding_box_x[1]:
                            number_of_not_shown_values += 1

                    ax.scatter(result_relative_profits[data_to_plot],
                               result_cum_loads,
                               c=matplotlib.colors.to_hex(scalar_map.to_rgba(j)),
                               marker="s",
                               label="{}".format(erf),
                               s=6, linewidths=.1, alpha=.8)

                    if i == 0:
                        leg = plt.legend(fontsize=14, markerscale=2, title="ERF", handletextpad=0, borderaxespad=0.175, borderpad=0.2)
                        for lh in leg.legendHandles:
                            lh.set_alpha(1.0)
                        plt.setp(leg.get_title(), fontsize=14)

            ax.set_xlim(bounding_box_x)
            ax.set_ylim(bounding_box_y)

            ax.tick_params(axis='both', which='major', labelsize=15.5)
            ax.tick_params(axis='x', which='minor', labelsize=15.5)
            plt.grid(True, which="both")

            if self.paper_mode:
                ax.set_title("Vanilla Rounding Performance", fontsize=17)
            else:
                title = "Vanilla Rounding Performance\n"
                #print observed_values_relative_profit
                title += "profit: min: {:.2f}; mean: {:.2f}; max: {:.2f}\n".format(np.nanmin(observed_values_relative_profit),
                                                                                    np.nanmean(observed_values_relative_profit),
                                                                                    np.nanmax(observed_values_relative_profit))
                title += "loads: min: {:.2f}; mean: {:.2f}; max: {:.2f}\n".format(np.nanmin(observed_values_load),
                                                                                   np.nanmean(observed_values_load),
                                                                                   np.nanmax(observed_values_load))

                title += "{} of {} points lie outside the displayed area".format(number_of_not_shown_values, len(observed_values_relative_profit))
                ax.set_title(title, fontsize=10)

            xlabel = "$\mathrm{Profit}({" + self.math_label_names[
                data_to_plot] + "}) / \mathrm{Profit}({\mathrm{MIP}_{\mathrm{MCF}}})$ [%]"
            ax.set_xlabel(xlabel, fontsize=16)
            ylabel = "$\mathrm{Max\,Load}\,({" + self.math_label_names[data_to_plot] + "})$ [%]"
            ax.set_ylabel(ylabel, fontsize=16)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%d"))

            self._show_and_or_save_plots(output_path, filename)

def _construct_filter_specs(scenario_parameter_space_dict, parameter_filter_keys, maxdepth=3):
    parameter_value_dic = dict()
    for parameter in parameter_filter_keys:
        _, parameter_values = extract_parameter_range(scenario_parameter_space_dict,
                                                      parameter)
        parameter_value_dic[parameter] = parameter_values
    # print parameter_value_dic.values()
    result_list = [None]
    for i in range(1, maxdepth + 1):
        for combi in combinations(parameter_value_dic, i):
            values = []
            for element_of_combi in combi:
                values.append(parameter_value_dic[element_of_combi])
            for v in product(*values):
                filter = []
                for (parameter, value) in zip(combi, v):
                    filter.append({'parameter': parameter, 'value': value})
                result_list.append(filter)

    return result_list

def evaluate_baseline_and_randround(dc_baseline,
                                    baseline_algorithm_id,
                                    baseline_execution_config,
                                    dc_randround,
                                    randround_algorithm_id,
                                    randround_execution_config,
                                    exclude_generation_parameters=None,
                                    parameter_filter_keys=None,
                                    show_plot=False,
                                    save_plot=True,
                                    overwrite_existing_files=True,
                                    forbidden_scenario_ids=None,
                                    papermode=True,
                                    maxdepthfilter=2,
                                    output_path="./",
                                    output_filetype="png"):
    """ Main function for evaluation, creating plots and saving them in a specific directory hierarchy.
    A large variety of plots is created. For heatmaps, a generic plotter is used while for general
    comparison plots (ECDF and scatter) an own class is used. The plots that shall be generated cannot
    be controlled at the moment but the respective plotters can be easily adjusted.

    :param dc_baseline: unpickled datacontainer of baseline experiments (e.g. MIP)
    :param baseline_algorithm_id: algorithm id of the baseline algorithm
    :param baseline_execution_config: execution config (numeric) of the baseline algorithm execution
    :param dc_randround: unpickled datacontainer of randomized rounding experiments
    :param randround_algorithm_id: algorithm id of the randround algorithm
    :param randround_execution_config: execution config (numeric) of the randround algorithm execution
    :param exclude_generation_parameters:   specific generation parameters that shall be excluded from the evaluation.
                                            These won't show in the plots and will also not be shown on axis labels etc.
    :param parameter_filter_keys:   name of parameters according to which the results shall be filtered
    :param show_plot:               Boolean: shall plots be shown
    :param save_plot:               Boolean: shall the plots be saved
    :param overwrite_existing_files:   shall existing files be overwritten?
    :param forbidden_scenario_ids:     list / set of scenario ids that shall not be considered in the evaluation
    :param papermode:                  nicely layouted plots (papermode) or rather additional information?
    :param maxdepthfilter:             length of filter permutations that shall be considered
    :param output_path:                path to which the results shall be written
    :param output_filetype:            filetype supported by matplotlib to export figures
    :return: None
    """

    if forbidden_scenario_ids is None:
        forbidden_scenario_ids = set()

    if exclude_generation_parameters is not None:
        for key, values_to_exclude in exclude_generation_parameters.iteritems():
            parameter_filter_path, parameter_values = extract_parameter_range(
                dc_baseline.scenario_parameter_container.scenarioparameter_room, key)

            parameter_dicts_baseline = lookup_scenario_parameter_room_dicts_on_path(
                dc_baseline.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)
            parameter_dicts_randround = lookup_scenario_parameter_room_dicts_on_path(
                dc_randround.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)

            for value_to_exclude in values_to_exclude:

                if value_to_exclude not in parameter_values:
                    raise RuntimeError("The value {} is not contained in the list of parameter values {} for key {}".format(
                        value_to_exclude, parameter_values, key
                    ))

                #add respective scenario ids to the set of forbidden scenario ids
                forbidden_scenario_ids.update(set(lookup_scenarios_having_specific_values(
                    dc_baseline.scenario_parameter_container.scenario_parameter_dict, parameter_filter_path, value_to_exclude)))

            #remove the respective values from the scenario parameter room such that these are not considered when
            #constructing e.g. axes
            parameter_dicts_baseline[-1][key] = [value for value in parameter_dicts_baseline[-1][key] if
                                                 value not in values_to_exclude]
            parameter_dicts_randround[-1][key] = [value for value in parameter_dicts_randround[-1][key] if
                                                  value not in values_to_exclude]


    if parameter_filter_keys is not None:
        filter_specs = _construct_filter_specs(dc_baseline.scenario_parameter_container.scenarioparameter_room,
                                               parameter_filter_keys,
                                               maxdepth=maxdepthfilter)
    else:
        filter_specs = [None]

    #initialize plotters

    baseline_plotter = SingleHeatmapPlotter(output_path=output_path,
                                            output_filetype=output_filetype,
                                            scenario_solution_storage=dc_baseline,
                                            algorithm_id=baseline_algorithm_id,
                                            execution_id=baseline_execution_config,
                                            heatmap_plot_type=HeatmapPlotType.Simple_MCF,
                                            show_plot=show_plot,
                                            save_plot=save_plot,
                                            overwrite_existing_files=overwrite_existing_files,
                                            forbidden_scenario_ids=forbidden_scenario_ids,
                                            paper_mode=papermode)

    randround_plotter = SingleHeatmapPlotter(output_path=output_path,
                                            output_filetype=output_filetype,
                                            scenario_solution_storage=dc_randround,
                                            algorithm_id=randround_algorithm_id,
                                            execution_id=randround_execution_config,
                                            heatmap_plot_type=HeatmapPlotType.Simple_RRT,
                                            show_plot=show_plot,
                                            save_plot=save_plot,
                                            overwrite_existing_files=overwrite_existing_files,
                                            forbidden_scenario_ids=forbidden_scenario_ids,
                                            paper_mode=papermode)

    comparison_plotter = ComparisonHeatmapPlotter(output_path=output_path,
                                                  output_filetype=output_filetype,
                                                  scenario_solution_storage=dc_baseline,
                                                  algorithm_id=baseline_algorithm_id,
                                                  execution_id=baseline_execution_config,
                                                  other_scenario_solution_storage=dc_randround,
                                                  other_algorithm_id=randround_algorithm_id,
                                                  other_execution_id=randround_execution_config,
                                                  heatmap_plot_type=HeatmapPlotType.Comparison_MCF_vs_RRT,
                                                  show_plot=show_plot,
                                                  save_plot=save_plot,
                                                  overwrite_existing_files=overwrite_existing_files,
                                                  forbidden_scenario_ids=forbidden_scenario_ids,
                                                  paper_mode=papermode)

    ecdf_capacity_violation_plotter = ComparisonBaselineVsRRT_Scatter_and_ECDF(output_path=output_path,
                                                                               output_filetype=output_filetype,
                                                                               baseline_solution_storage=dc_baseline,
                                                                               baseline_algorithm_id=baseline_algorithm_id,
                                                                               baseline_execution_id=baseline_execution_config,
                                                                               randround_solution_storage=dc_randround,
                                                                               randround_algorithm_id=randround_algorithm_id,
                                                                               randround_execution_id=randround_execution_config,
                                                                               show_plot=show_plot,
                                                                               save_plot=save_plot,
                                                                               overwrite_existing_files=overwrite_existing_files,
                                                                               forbidden_scenario_ids=forbidden_scenario_ids,
                                                                               paper_mode=papermode)

    plotters = [ecdf_capacity_violation_plotter, baseline_plotter, randround_plotter, comparison_plotter]

    for filter_spec in filter_specs:

        for plotter in plotters:
            plotter.plot_figure(filter_spec)
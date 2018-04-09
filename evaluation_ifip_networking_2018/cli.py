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
import sys

import click
import pickle
import logging

import alib.cli
from alib import run_experiment, util
from vnep_approx import modelcreator_ecg_decomposition

from . import randomized_rounding
from . import evaluation
from . import plot_data as pd

@click.group()
def cli():
    pass

def initialize_logger(filename, log_level_print, log_level_file, allow_override=False):
    print log_level_print.upper()
    print log_level_file.upper()
    log_level_print = logging._levelNames[log_level_print.upper()]
    log_level_file = logging._levelNames[log_level_file.upper()]
    util.initialize_root_logger(filename, log_level_print, log_level_file, allow_override=allow_override)

@cli.command()
@click.argument('pickle_file', type=click.File('r'))
@click.option('--col_output_limit', default=None)
def pretty_print(pickle_file, col_output_limit):
    data = pickle.load(pickle_file)
    pp = util.PrettyPrinter()
    print pp.pprint(data, col_output_limit=col_output_limit)

@cli.command()
@click.argument('scenario_output_file')
@click.argument('parameters', type=click.File('r'))
@click.option('--threads', default=1)
def generate_scenarios(scenario_output_file, parameters, threads):
    alib.cli.f_generate_scenarios(scenario_output_file, parameters, threads)


@cli.command()
@click.argument('experiment_yaml', type=click.File('r'))
@click.argument('min_scenario_index', type=click.INT)
@click.argument('max_scenario_index', type=click.INT)
@click.option('--concurrent', default=1)
@click.option('--log_level_print', type=click.STRING, default="info")
@click.option('--log_level_file', type=click.STRING, default="debug")
def start_experiment(experiment_yaml,
                     min_scenario_index, max_scenario_index,
                     concurrent, log_level_print, log_level_file):
    click.echo('Start Experiment')
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(experiment_yaml.name).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_experiment_execution.log".format(file_basename))
    initialize_logger(log_file, log_level_print, log_level_file)

    run_experiment.register_algorithm(
        modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition.ALGORITHM_ID,
        modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition
    )

    run_experiment.register_algorithm(
        randomized_rounding.RandomizedRoundingTriumvirat.ALGORITHM_ID,
        randomized_rounding.RandomizedRoundingTriumvirat
    )

    run_experiment.run_experiment(
        experiment_yaml,
        min_scenario_index, max_scenario_index,
        concurrent
    )


@cli.command()
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None)
@click.option('--log_level_print', type=click.STRING, default="info")
@click.option('--log_level_file', type=click.STRING, default="debug")
def reduce_baseline_pickle(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "reduce_{}.log".format(os.path.basename(input_pickle_file)))
    initialize_logger(log_file, log_level_print, log_level_file)
    reducer = pd.BaselineResultReducer()
    reducer.reduce_baseline_solution(input_pickle_file, output_pickle_file)


@cli.command()
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None)
@click.option('--log_level_print', type=click.STRING, default="info")
@click.option('--log_level_file', type=click.STRING, default="debug")
def reduce_randround_pickle(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "reduce_{}.log".format(os.path.basename(input_pickle_file)))
    initialize_logger(log_file, log_level_print, log_level_file)
    reducer = pd.RandRoundResultReducer()
    reducer.reduce_randomized_rounding_solution(input_pickle_file, output_pickle_file)

def collect_existing_alg_ids(execution_parameter_container):
    list_of_alg_ids = []
    for alg_dict in execution_parameter_container.algorithm_parameter_list:
        if alg_dict['ALG_ID'] not in list_of_alg_ids:
            list_of_alg_ids.append(alg_dict['ALG_ID'])
    return list_of_alg_ids

def query_algorithm_id_and_execution_id(logger,
                                        pickle_name,
                                        execution_parameter_container,
                                        algorithm_id,
                                        execution_config_id,
                                        query_even_when_only_one_option=False):

    list_of_alg_ids = collect_existing_alg_ids(execution_parameter_container)
    if algorithm_id is not None and algorithm_id not in list_of_alg_ids:
        logger.error("The provided algorithm id {} for the pickle {} is not contained in the contained list of algorithm ids: {}.".format(algorithm_id, pickle_name, list_of_alg_ids))
        algorithm_id=None
    if algorithm_id is None:
        if len(list_of_alg_ids) == 0:
            raise RuntimeError("It seems that the pickle {} does not contain any algorithm information. Abort.".format(pickle_name))
        if not query_even_when_only_one_option and len(list_of_alg_ids) == 1:
            algorithm_id = list_of_alg_ids[0]
            logger.info(
                " .. selected algorithm id '{}' for the pickle {}".format(algorithm_id, pickle_name))
        else:
            logger.info("\nAvailable algorithm ids for the pickle {} are: {}".format(pickle_name, list_of_alg_ids))
            algorithm_id = click.prompt("Please select an algorithm id for the pickle {}:".format(pickle_name), type=click.Choice(list_of_alg_ids))

    list_of_suitable_execution_ids = [x for x in execution_parameter_container.get_execution_ids(ALG_ID=algorithm_id)]
    if execution_config_id is not None and execution_config_id not in list_of_suitable_execution_ids:
        logger.error(
            "The provided execution id {} for the algorithm id {} for the pickle {} is not contained in the contained list of algorithm ids: {}.".format(
                execution_config_id, algorithm_id, pickle_name, list_of_alg_ids))
        execution_config_id=None
    if execution_config_id is None:
        if len(list_of_suitable_execution_ids) == 0:
            raise RuntimeError(
                "It seems that the pickle {} does not contain any suitable execution ids for algorithm id {}. Abort.".format(
                    pickle_name, algorithm_id))
        if not query_even_when_only_one_option and len(list_of_suitable_execution_ids) == 1:
            execution_config_id = list_of_suitable_execution_ids[0]
            logger.info(
                " .. selected execution id '{}' for the pickle {} as it is the only one for algorithm id {}".format(execution_config_id, pickle_name, algorithm_id))
        else:
            logger.info("\nAvailable execution ids for the algorithm id {} of the pickle {} are...".format(algorithm_id,
                                                                                                           pickle_name,
                                                                                                           list_of_alg_ids))
            for execution_id in list_of_suitable_execution_ids:
                logger.info(
                    "\nExecution id {} corresponds to {}".format(execution_id,execution_parameter_container.algorithm_parameter_list[execution_id]))


            execution_config_id = click.prompt("Please select an execution id for the algorithm id {} for the pickle {}:".format(algorithm_id,
                                                                                                                                 pickle_name),
                                               type=click.Choice([str(x) for x in list_of_suitable_execution_ids]))
            execution_config_id = int(execution_config_id)

    return algorithm_id, execution_config_id

@cli.command()
@click.argument('baseline_pickle_name', type=click.Path())
@click.option('--baseline_algorithm_id', type=click.STRING, default=None)
@click.option('--baseline_execution_config', type=click.INT, default=None)
@click.argument('randround_pickle_name', type=click.Path())
@click.option('--randround_algorithm_id', type=click.STRING, default=None)
@click.option('--randround_execution_config', type=click.INT, default=None)
@click.argument('output_directory', type=click.Path())
@click.option('--overwrite/--no_overwrite', default=True)
@click.option('--papermode/--non-papermode', default=True)
@click.option('--output_filetype', type=click.Choice(['png', 'pdf', 'eps']), default="png")
@click.option('--log_level_print', type=click.STRING, default="info")
@click.option('--log_level_file', type=click.STRING, default="debug")
def evaluate_results(baseline_pickle_name,
                     baseline_algorithm_id,
                     baseline_execution_config,
                     randround_pickle_name,
                     randround_algorithm_id,
                     randround_execution_config,
                     output_directory,
                     overwrite,
                     papermode,
                     output_filetype,
                     log_level_print,
                     log_level_file):

    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "evaluate_pickles_{}_{}.log".format(os.path.basename(baseline_pickle_name),
                                                                os.path.basename(randround_pickle_name)))
    initialize_logger(log_file, log_level_print, log_level_file, allow_override=True)

    baseline_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, baseline_pickle_name)
    randround_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, randround_pickle_name)

    #get root logger
    logger = logging.getLogger()

    logger.info("Reading reduced baseline pickle at {}".format(baseline_pickle_path))
    baseline_results = None
    with open(baseline_pickle_path, "rb") as f:
        baseline_results = pickle.load(f)

    logger.info("Reading reduced randround pickle at {}".format(randround_pickle_path))
    randround_results = None
    with open(randround_pickle_path, "rb") as f:
        randround_results = pickle.load(f)

    logger.info("Loading algorithm identifiers and execution ids..")

    baseline_algorithm_id, baseline_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                           baseline_pickle_name,
                                                                                           baseline_results.execution_parameter_container,
                                                                                           baseline_algorithm_id,
                                                                                           baseline_execution_config)

    randround_algorithm_id, randround_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                           randround_pickle_name,
                                                                                           randround_results.execution_parameter_container,
                                                                                           randround_algorithm_id,
                                                                                           randround_execution_config)

    evaluation.LOOKUP_FUNCTION_BASELINE = (lambda x: x[baseline_algorithm_id][baseline_execution_config])
    evaluation.LOOKUP_FUNCTION_RANDROUND = (lambda x: x[randround_algorithm_id][randround_execution_config])


    output_directory = os.path.normpath(output_directory)

    logger.info("Setting output path to {}".format(output_directory))
    evaluation.OUTPUT_PATH = output_directory
    evaluation.OUTPUT_FILETYPE = output_filetype



    logger.info("Starting evaluation...")
    evaluation.plot_heatmaps(baseline_results,
                             randround_results,
                             maxdepthfilter=0,
                             overwrite_existing_files=(overwrite),
                             output_path=output_directory,
                             output_filetype=output_filetype,
                             papermode=papermode)




if __name__ == '__main__':
    cli()

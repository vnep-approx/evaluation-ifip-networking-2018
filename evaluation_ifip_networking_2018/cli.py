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

import click
import pickle
import logging

import alib.cli
from alib import run_experiment, util
from vnep_approx import modelcreator_ecg_decomposition, randomized_rounding_triumvirate

from . import evaluation
from . import plot_data as pd

@click.group()
def cli():
    """
    This command-line interface allows you to access major parts of the VNEP-Approx framework
    developed by Matthias Rost, Elias Döhne, Alexander Elvers, and Tom Koch.
    In particular, it allows to reproduce the results presented in the paper:

    "Virtual Network Embedding Approximations: Leveraging Randomized Rounding", Matthias Rost and Stefan Schmid, IFIP Networking 2018.

    Note that each commands provides a detailed help page. To access the help, simply type the commmand and --help.

    """
    pass

def initialize_logger(filename, log_level_print, log_level_file, allow_override=False):
    log_level_print = logging._levelNames[log_level_print.upper()]
    log_level_file = logging._levelNames[log_level_file.upper()]
    util.initialize_root_logger(filename, log_level_print, log_level_file, allow_override=allow_override)

@cli.command(short_help="pretty print contents of pickle file")
@click.argument('pickle_file', type=click.File('r'))
@click.option('--col_output_limit', default=None, help="The number of items that shall be printed.")
def pretty_print(pickle_file, col_output_limit):
    """
        Pretty print pickle file (absolute path).
        Note that, if the pickle file is large, it may take some time to generate the output.
        Furthermore, a high amount of RAM might be necessary.
    """
    data = pickle.load(pickle_file)
    pp = util.PrettyPrinter()
    print pp.pprint(data, col_output_limit=col_output_limit)

@cli.command(short_help="generate scenarios according to yaml specification")
@click.argument('yaml_parameter_file', type=click.File('r'))
@click.argument('scenario_output_file')
@click.option('--threads', default=1, help="Number of processed to be used for generating the scenarios.")
def generate_scenarios(yaml_parameter_file, scenario_output_file, threads):
    """ Generate scenarios according to yaml_parameter_file. Note that while the yaml_parameter_file can be placed anywhere,
        the resuling scenario_output_file will be placed into ALIB_EXPERIMENT_HOME/output.
        Accordingly, the environment variable ALIB_EXPERIMENT_HOME must be set.
        The process of scenario generation is logged into ALIB_EXPERIMENT_HOME/log
    """
    alib.cli.f_generate_scenarios(scenario_output_file, yaml_parameter_file, threads)


@cli.command(short_help="compute solutions to scenarios")
@click.argument('experiment_yaml', type=click.File('r'))
@click.argument('min_scenario_index', type=click.INT)
@click.argument('max_scenario_index', type=click.INT)
@click.option('--concurrent', default=1, help="number of processes to be used in parallel")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
@click.option('--shuffle_instances/--original_order', default=True, help="shall instances be shuffled or ordered according to their ids (ascendingly)")
@click.option('--overwrite_existing_temporary_scenarios/--use_existing_temporary_scenarios', default=False, help="shall existing temporary scenario files be overwritten or used?")
@click.option('--overwrite_existing_intermediate_solutions/--use_existing_intermediate_solutions', default=False, help="shall existing intermediate solution files be overwritten or used?")
def start_experiment(experiment_yaml,
                     min_scenario_index, max_scenario_index,
                     concurrent,
                     log_level_print,
                     log_level_file,
                     shuffle_instances,
                     overwrite_existing_temporary_scenarios,
                     overwrite_existing_intermediate_solutions):
    """ Execute experiments according to given experiment_yaml file (absolute path).
        The contents of the experiment_yaml detail which scenario file to load which must be
        located in ALIB_EXPERIMENT_HOME/input. The min_scenario_index and max_scenario_index
        specify the range of scenario ids to consider: only scenarios of ids in
        [min_scenario_index, max_scenario_index] are considered.
        The --concurrent option can be used to solve multiple scenarios in parallel. Note
        that this does not equal the number of threads as e.g. Gurobi might use more than
        one thread (per solving process).

        The logs are stored in ALIB_EXPERIMENT_HOME.

        The environment variable ALIB_EXPERIMENT_HOME needs to be set!
    """
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
        randomized_rounding_triumvirate.RandomizedRoundingTriumvirate.ALGORITHM_ID,
        randomized_rounding_triumvirate.RandomizedRoundingTriumvirate
    )

    run_experiment.run_experiment(
        experiment_yaml,
        min_scenario_index, max_scenario_index,
        concurrent,
        shuffle_instances,
        overwrite_existing_temporary_scenarios,
        overwrite_existing_intermediate_solutions
    )


@cli.command(short_help="extracts data to be plotted for baseline (MCF)")
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None, help="file to write to")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
def reduce_to_plotdata_baseline_pickle(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    """ Given a scenario solution pickle (input_pickle_file) this function extracts data
        to be plotted and writes it to --output_pickle_file. If --output_pickle_file is not
        given, a default name (derived from the input's basename) is derived.

        The input_file must be contained in ALIB_EXPERIMENT_HOME/input and the output
        will be written to ALIB_EXPERIMENT_HOME/output while the log is saved in
        ALIB_EXPERIMENT_HOME/log.
    """
    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "reduce_{}.log".format(os.path.basename(input_pickle_file)))
    initialize_logger(log_file, log_level_print, log_level_file)
    reducer = pd.BaselineResultReducer()
    reducer.reduce_baseline_solution(input_pickle_file, output_pickle_file)


@cli.command(short_help="extracts data to be plotted for randomized rounding alg (Triumvirate)")
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None)
@click.option('--log_level_print', type=click.STRING, default="info")
@click.option('--log_level_file', type=click.STRING, default="debug")
def reduce_to_plotdata_randround_pickle(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    """ Given a scenario solution pickle (input_pickle_file) for randomized rounding, this          function extracts data  to be plotted and writes it to --output_pickle_file.
        If --output_pickle_file is not given, a default name (derived from the input's              basename) is derived.

        The input_file must be contained in ALIB_EXPERIMENT_HOME/input and the output
        will be written to ALIB_EXPERIMENT_HOME/output while the log is saved in
        ALIB_EXPERIMENT_HOME/log.
    """
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

@cli.command(short_help="create plots for baseline and randround solution")
@click.argument('baseline_pickle_name', type=click.Path())      #pickle in ALIB_EXPERIMENT_HOME/input storing baseline results
@click.argument('randround_pickle_name', type=click.Path())     #pickle in ALIB_EXPERIMENT_HOME/input storing randround results
@click.argument('output_directory', type=click.Path())          #path to which the result will be written
@click.option('--baseline_algorithm_id', type=click.STRING, default=None, help="algorithm id of baseline algorithm; if not given it will be asked for.")
@click.option('--baseline_execution_config', type=click.INT, default=None, help="execution (configuration) id of baseline alg; if not given it will be asked for.")
@click.option('--randround_algorithm_id', type=click.STRING, default=None, help="algorithm id of randround algorithm; if not given it will be asked for.")
@click.option('--randround_execution_config', type=click.INT, default=None, help="execution (configuration) id of randround alg; if not given it will be asked for.")
@click.option('--exclude_generation_parameters', type=click.STRING, default=None, help="generation parameters that shall be excluded. "
                                                                                       "Must ge given as python evaluable list of dicts. "
                                                                                       "Example format: \"{'number_of_requests': [20]}\"")
@click.option('--filter_parameter_keys', type=click.STRING, default=None, help="generation parameters whose values will represent filters. "
                                                                               "Must be given as string detailing a python list containing strings."
                                                                               "Example: \"['number_of_requests', 'edge_resource_factor', 'node_resource_factor']\"")
@click.option('--filter_max_depth', type=click.INT, default=0, help="Maximal recursive depth up to which permutations of filters are considered.")
@click.option('--overwrite/--no_overwrite', default=True, help="overwrite existing files?")
@click.option('--papermode/--non-papermode', default=True, help="output 'paper-ready' figures or figures containing additional statistical data?")
@click.option('--output_filetype', type=click.Choice(['png', 'pdf', 'eps']), default="png", help="the filetype which shall be created")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for stdout")
def evaluate_results(baseline_pickle_name,
                     randround_pickle_name,
                     output_directory,
                     baseline_algorithm_id,
                     baseline_execution_config,
                     randround_algorithm_id,
                     randround_execution_config,
                     exclude_generation_parameters,
                     filter_parameter_keys,
                     filter_max_depth,
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

    output_directory = os.path.normpath(output_directory)

    logger.info("Setting output path to {}".format(output_directory))
    evaluation.OUTPUT_PATH = output_directory
    evaluation.OUTPUT_FILETYPE = output_filetype


    if exclude_generation_parameters is not None:
        exclude_generation_parameters = eval(exclude_generation_parameters)
    if filter_parameter_keys is not None:
        filter_parameter_keys = eval(filter_parameter_keys)

    logger.info("Starting evaluation...")
    evaluation.evaluate_baseline_and_randround(baseline_results,
                                               baseline_algorithm_id,
                                               baseline_execution_config,
                                               randround_results,
                                               randround_algorithm_id,
                                               randround_execution_config,
                                               exclude_generation_parameters=exclude_generation_parameters,
                                               parameter_filter_keys=filter_parameter_keys,
                                               maxdepthfilter=filter_max_depth,
                                               overwrite_existing_files=(overwrite),
                                               output_path=output_directory,
                                               output_filetype=output_filetype,
                                               papermode=papermode)




if __name__ == '__main__':
    cli()

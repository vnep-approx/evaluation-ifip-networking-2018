#!/bin/bash

#either the $PYTHONPATH must be set to include alib, vnep_approx and evaluation_ifip_networking_2018 or 
#you execute this from within the virtual environment in which these packages were installed

export ALIB_EXPERIMENT_HOME=$(pwd)

mkdir -p log/ && mkdir -p input && mkdir -p output

function move_logs_and_output() {
	mkdir -p old_logs
	echo "make sure to check the logs for errors and that the generated output is correct"
	mv output/* input/
	mv log/* old_logs/
}

#generate scenarios
python -m evaluation_ifip_networking_2018.cli generate_scenarios sample_scenarios.yml sample_scenarios.pickle
move_logs_and_output

#run multi-commodity flow mixed-integer program
python -m evaluation_ifip_networking_2018.cli start_experiment sample_mip_execution.yml 0 10000 --concurrent 2
move_logs_and_output

#run randomized rounding algorithms
python -m evaluation_ifip_networking_2018.cli start_experiment sample_randround_execution.yml 0 10000 --concurrent 2
move_logs_and_output

#extract data to be plotted
python -m evaluation_ifip_networking_2018.cli reduce_to_plotdata_baseline_pickle sample_scenarios_results_mip.pickle 
move_logs_and_output
python -m evaluation_ifip_networking_2018.cli reduce_to_plotdata_randround_pickle sample_scenarios_results_randround.pickle
move_logs_and_output

#generate plots in folder ./plots
mkdir -p ./plots
python -m evaluation_ifip_networking_2018.cli evaluate_results sample_scenarios_results_mip_reduced.pickle sample_scenarios_results_randround_reduced.pickle ./plots --overwrite --output_filetype png --non-papermode  --filter_max_depth 0





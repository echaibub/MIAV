This repository contains the code for the paper: Chaibub Neto (2026) *Using maximal information auxiliary variables to improve synthetic data generation based on TabPFN foundation models*, ICLR, 2026.

It contains a mix of Python scripts, Jupyter notebooks, and R scripts implemeting the MIAV, JF, and FC synthetic data generation strategies based on TabPFN models (and TabICL models for categorical data), alongside code for running the experiments and evaluating the different synthesizer's performances.

The synthetic data generation was performed mostly in Python, while the performance evaluations were performed in R. 

See below brief descriptions of each file.

# Python code

"utility_functions_implementing_tabpfn_generators_iclr.py": 
Python script containing utility functions implementing the generators.

"additional_utility_functions_for_tabpfn_generators_iclr.py":
Python script with additional helper functions.

"generate_synthetic_data_for_real_data_experiments_openml_cc18.ipynb":
Jupyter notebook for performing synthetic data generation on the first set of 21 datasets from the OpenML-CC18 benchmark suite (listed in Table 4). 

"generate_synthetic_data_for_real_data_experiments_openml_cc18_additional.ipynb":
Jupyter notebook for performing synthetic data generation on the additional 15 datasets from the OpenML-CC18 benchmark suite (listed in Table 4).

"generate_synthetic_data_for_real_data_experiments_openml_cc18_categorical.ipynb":
Jupyter notebook for performing synthetic data generation (using TabICL-based and TabPFN-based strategies) on the 8 categorical datasets from the OpenML-CC18 benchmark suite listed in Table 8.

"myav_jf_fc_on_baseline_data.ipynb":
Jupyter notebook for performing MIAV, JF, and FC synthetic data generation on the baseline comparison datasets listed in Table 5. 

"noisy_miav_for_baseline_comparisons.ipynb":
Jupyter notebook for performing noisy-MIAV synthetic data generation on the baseline comparison datasets listed in Table 5.

"synthcity_plugin_for_smotenc_generator.py":
Synthcity plugin implemeting the SMOTE generator.

"synthcity_baseline_comparisons.ipynb":
Jupyter notebook for performing synthetic data generation on the baseline comparison datasets using the DDPM, ARF, CTGAN, TVAE, and bayesian-network generators (using the Synthcity library).

"runtime_benchmarking_experiments.ipynb":
Jupyter notebook for running the runtime benchmark experiments comparing the MIAV, JF, and FC strategies.

"generate_tabpfn_predictions.py":
Python script for running TabPFN in R using reticulate. To run TabPFN in R you need to first create a virtual environment (venv), activate it, and install  TabPFN using a terminal, and then download the above file in the virtual environment and run:

library(reticulate)

use_virtualenv("~/folder where you created your virtual environment/venv")

source_python("generate_tabpfn_predictions.py")

# R code

"utility_functions_for_miav_tabpfn_iclr.R":
Script containing utility functions implementing the generators, evaluation metrics, and the helpers for performing the experiments. 

"run_simulated_data_experiments_abs_rho.R":
Script for running the simulated data experiments based on correlated beta distributed data.

"run_real_world_data_evaluations_on_first_21_datasets.R" and
"run_real_world_data_evaluations_on_additional_15_datasets.R":
Scripts for running the data fidelity and data privacy evaluations on the real-world data from the OpenML-CC18 benchmarck suite. The first script perfoms the evaluations on a first set of datasets containing less than 2000 samples and less than 100 features, while the second one run the evaluations of a second set containing 15 additional datasets (containg less 
than 10000 samples and 500 features, and which were not included in the first set). The 36 datasets are listed in Table 4.

"run_real_world_data_evaluations_on_baseline_datasets.R":
Script for running the data fidelity and data privacy evaluations on the 7 real-world datasets listed in Table 5.

"run_real_world_data_evaluations_on_categorical_datasets.R":
Script for running the data fidelity evaluations on the the 8 categorical real-world datasets listed in Table 8.

"generate_illustrative_figures.R":
Script for generating Figures 1, 2, and 3 in the main text, Figures 5, 6, 7, 8, and 9 in Appendix D, and Figure 10 in Appendix E.

"generate_manuscript_experiment_results_figures.R":
Script for generating Figure 4 in the main text, Figures 14, 15, 16, 17, 18, and 19 in Appendix I.7, and Figures 23 and 24 in Appendix K.

"generate_runtime_experiments_figures.R":
Script for generating Figures 11 and 12 in Appendix H.2.

"mle_efficiency_rebuttal.R":
Script for running the ML efficiency analyses on the simulated data and the 43 real-world datasets (21 + 15 + 7).

"wasserstein_distance_rebuttal.R":
Script for running the Wasserstein distance analyses on the simulated data and the 43 real-world datasets (21 + 15 + 7).

"generate_additional_metrics_experimental_results_figure.R":
Script for generating Figure 13 in Appendix I.6.

"generate_plots_for_noisy_miav_comparisons.R":
Script for generating Figures 18, 19, and 20 in Appendix J.

"compare_miav_under_uniform_gaussian_exponential_noise.R":
Script implementing the analyses in Appendix L.

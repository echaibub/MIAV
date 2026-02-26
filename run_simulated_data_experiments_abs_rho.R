
# source utility functions
source("utility_functions_for_miav_tabpfn_iclr.R")


# To run TabPFN in R: 
# First in a terminal, create a virtual environment, activate it, and install 
# TabPFN. Then download the "generate_tabpfn_predictions.py" script.
#
# In R load reticulate and run the python script, which make classifiers 
# and regression models based on TabPFN available in R.
library(reticulate)
use_virtualenv("~/TabPFN/venv")
source_python("~/TabPFN/venv/generate_tabpfn_predictions.py")


######################################################
######################################################
######################################################

n_sim <- 10
n <- 400

percent_grid <- seq(0.05, 0.3, by = 0.05)

set.seed(123)
out_0.95 <- RunSimulations(n_sim = n_sim, n = n, abs_rho = 0.95, my_seed = 1001,
                           percent_grid = percent_grid)
save(out_0.95, file = "simulation_outputs_abs_rho_0.95.RData", compress = TRUE)

set.seed(123)
out_0.75 <- RunSimulations(n_sim = n_sim, n = n, abs_rho = 0.75, my_seed = 1002,
                           percent_grid = percent_grid)
save(out_0.75, file = "simulation_outputs_abs_rho_0.75.RData", compress = TRUE)

set.seed(123)
out_0.5 <- RunSimulations(n_sim = n_sim, n = n, abs_rho = 0.5, my_seed = 1003,
                          percent_grid = percent_grid)
save(out_0.5, file = "simulation_outputs_abs_rho_0.5.RData", compress = TRUE)

set.seed(123)
out_0.25 <- RunSimulations(n_sim = n_sim, n = n, abs_rho = 0.25, my_seed = 1004,
                           percent_grid = percent_grid)
save(out_0.25, file = "simulation_outputs_abs_rho_0.25.RData", compress = TRUE)

set.seed(123)
out_0 <- RunSimulations(n_sim = n_sim, n = n, abs_rho = 0, my_seed = 1005,
                        percent_grid = percent_grid)
save(out_0, file = "simulation_outputs_abs_rho_0.RData", compress = TRUE)



source("utility_functions_for_miav_tabpfn_iclr.R")

##############################################################################
## additional utility functions implementing the Wasserstein distance 
##############################################################################

library(transport)
library(Rfast)

# compute the squared Euclidean cost matrix
SquaredEuclideanCostMatrix <- function(X, Y) {
  C <- Rfast::dista(X, Y, square = TRUE)
  
  return(C)
}


MyScaling <- function(x, alpha = 0) {
  # Check if data has no variability
  num_unique <- length((unique(x)))
  if (num_unique == 1) {
    # set variables with no variability to 0.5
    # (don't want to drop because sometimes the issue
    # appears only on the synthetic data)
    y <- rep(0.5, length(x)) 
  }
  else {
    aux <- quantile(x, probs = c(alpha, 1 - alpha), na.rm = TRUE)
    x_1 <- aux[1]
    x_2 <- aux[2]
    y <- (x - x_1)/(x_2 - x_1)
  }
  return(y)
}


WassersteinDistance <- function(dat1_n, dat2_n) {
  ## apply min-max scaling to datasets 1 and 2
  dat1_n <- MyScaling(dat1_n, alpha = 0)
  dat2_n <- MyScaling(dat2_n, alpha = 0)

  # set the same weight to each datapoint
  n <- nrow(dat1_n)
  a <- rep(1/n, n)
  n <- nrow(dat2_n)
  b <- rep(1/n, n)

  # compute the cost matrix
  costm <- SquaredEuclideanCostMatrix(dat1_n, dat2_n)
  
  # compute the Wasserstein distance
  WD <- wasserstein(a = a, b = b, costm = costm, p = 2)
  
  return(WD)
}


EvaluateSyntheticDataWD <- function(df_split,
                                    df_synth) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  wd <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(wd) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
  ## n_datasets
  for (i in seq(n_datasets)) {
    
    for (j in seq(n_splits)) {
      cat(c(i, j), "\n")
      
      dat_orig <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "orig")
      
      dat_synt <- GrabDataset(df = df_synth, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "syn")
      
      # If data synthesis failed for a dataset (so that it has 0 rows),
      # we skip the evaluation
      if (nrow(dat_synt) > 0) {
        ## add "X" to variable names
        dat_orig <- data.frame(dat_orig)
        dat_synt <- data.frame(dat_synt)
        
        aux_type <- GetVariableTypes(dat_orig)
        num_variables <- aux_type$num_variables
        
        cat("wd ", c(i, j), "\n") 
        tmp <- try(WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                       dat2_n = dat_synt[, num_variables]),
                   silent = TRUE)
        if (!inherits(tmp, "try-error")) {
          wd[j, i] <- tmp
        }
      }
    }
  }
  
  return(list(wd = wd))
}


EvaluateHoldoutDataWD <- function(df_split) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  wd <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(wd) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
  ## n_datasets
  for (i in seq(n_datasets)) {
    
    for (j in seq(n_splits)) {
      cat(c(i, j), "\n")
      
      dat_orig <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "orig")
      
      dat_hold <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "hold")
      
      dat_orig <- data.frame(dat_orig)
      dat_hold <- data.frame(dat_hold)
      
      aux_type <- GetVariableTypes(dat_orig)
      num_variables <- aux_type$num_variables
      
      cat("wd ", c(i, j), "\n") 
      tmp <- try(WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                     dat2_n = dat_hold[, num_variables]),
                 silent = TRUE)
      if (!inherits(tmp, "try-error")) {
        wd[j, i] <- tmp
      }
    }
  }
  
  return(list(wd = wd))
}


EvaluateSmoteDataWD <- function(df_split, k = 5) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  wd <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(wd) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
  ## n_datasets
  for (i in seq(n_datasets)) {
    
    for (j in seq(n_splits)) {
      cat(c(i, j), "\n")
      
      dat_orig <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "orig")
      
      ## generate synthetic data using smote
      dat_synt <- SynthSMOTENC(dat_orig, k = k)
      
      dat_orig <- data.frame(dat_orig)
      dat_synt <- data.frame(dat_synt)
      
      aux_type <- GetVariableTypes(dat_orig)
      num_variables <- aux_type$num_variables
      
      cat("wd ", c(i, j), "\n") 
      tmp <- try(WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                     dat2_n = dat_synt[, num_variables]),
                 silent = TRUE)
      if (!inherits(tmp, "try-error")) {
        wd[j, i] <- tmp
      }
    }
  }
  
  return(list(wd = wd))
}


EvaluateSyntheticDataBaselineWD <- function(df_split,
                                            df_synth) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  wd <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(wd) <- dataset_names
  
  for (j in seq(n_splits)) {
    cat(c(j), "\n")
    
    dat_orig <- GrabDataset(df = df_split, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "orig")
    
    dat_synt <- GrabDataset(df = df_synth, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "syn")
    
    dat_orig <- data.frame(dat_orig)
    dat_synt <- data.frame(dat_synt)
    
    aux_type <- GetVariableTypes(dat_orig)
    num_variables <- aux_type$num_variables
    
    cat("wd ", j, "\n")
    tmp <- try(WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                   dat2_n = dat_synt[, num_variables]),
               silent = TRUE)
    if (!inherits(tmp, "try-error")) {
      wd[j, 1] <- tmp
    }
  }
  
  return(list(wd = wd))
}


EvaluateHoldoutDataBaselineWD <- function(df_split) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  wd <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(wd) <- dataset_names
  
  for (j in seq(n_splits)) {
    cat(c(j), "\n")
    
    dat_orig <- GrabDataset(df = df_split, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "orig")
    
    dat_hold <- GrabDataset(df = df_split, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "hold")
    
    dat_orig <- data.frame(dat_orig)
    dat_hold <- data.frame(dat_hold)
    
    aux_type <- GetVariableTypes(dat_orig)
    num_variables <- aux_type$num_variables
    
    cat("wd ", j, "\n")
    tmp <- try(WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                   dat2_n = dat_hold[, num_variables]),
               silent = TRUE)
    if (!inherits(tmp, "try-error")) {
      wd[j, 1] <- tmp
    }
  }
  
  return(list(wd = wd))
}


EvaluateSmoteDataBaselineWD <- function(df_split, k = 5) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  wd <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(wd) <- dataset_names
  
  for (j in seq(n_splits)) {
    cat(c(j), "\n")
    
    dat_orig <- GrabDataset(df = df_split, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "orig")
    
    ## generate synthetic data using smote
    dat_synt <- SynthSMOTENC(dat_orig, k = k)
    
    dat_orig <- data.frame(dat_orig)
    dat_synt <- data.frame(dat_synt)
    
    aux_type <- GetVariableTypes(dat_orig)
    num_variables <- aux_type$num_variables
    
    cat("wd ", j, "\n")
    tmp <- try(WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                   dat2_n = dat_synt[, num_variables]),
               silent = TRUE)
    if (!inherits(tmp, "try-error")) {
      wd[j, 1] <- tmp
    }
  }
  
  return(list(wd = wd))
}


RunEvaluationsBaselineWD <- function(ds_name,
                                     data_path,
                                     generator_names) {
  df_split <- read_feather(paste0(data_path, paste0(ds_name, "_orig_hold_splits.feather")))
  
  # Create list containing generator names
  n_generators <- length(generator_names)
  df_generators <- vector(mode = "list", n_generators)
  names(df_generators) <- generator_names
  for (i in seq(n_generators)) {
    fname <- paste0(data_path, paste0(ds_name, "_syn_", generator_names[i], ".feather"))
    df_generators[[i]] <- read_feather(fname)
  }
  
  # Run evaluations
  aux_list <- vector(mode = "list", length = n_generators + 1)
  names(aux_list) <- c("holdout", generator_names)
  
  cat("running evaluations on the holdout set", "\n")
  aux_list[[1]] <- EvaluateHoldoutDataBaselineWD(df_split = df_split)
  
  for (i in seq(n_generators)) {
    cat("running evaluations for: ", generator_names[i], "\n")
    aux_list[[i + 1]] <- EvaluateSyntheticDataBaselineWD(df_split = df_split,
                                                         df_synth = df_generators[[i]])
  }
  
  # Organize outputs by evaluation metric
  p <- length(aux_list)
  nms <- names(aux_list)
  
  wd <- aux_list[[1]]$wd
  for (i in seq(2, p)) {
    wd <- cbind(wd, aux_list[[i]]$wd)
  }
  colnames(wd) <- nms
  
  return(list(wd = wd))
}


RunSimulationsWD <- function(n_sim, n, abs_rho, my_seed) {
  
  sim_seeds <- NULL
  if (!is.null(my_seed)) {
    set.seed(my_seed)
    sim_seeds <- sample(seq(1e+4, 1e+5), n_sim, replace = FALSE)
  }

  methods_names <- c("hold", "jf", "fc", "miav", "smote")
  
  wd <- matrix(NA, n_sim, length(methods_names)) 
  colnames(wd) <- methods_names
  
  for (i in seq(n_sim)) {
    
    cat("######################################### run simulation ", i, "\n")
    
    if (!is.null(sim_seeds)) {
      set.seed(sim_seeds[i])
    }
    
    beta_pars_list <- list()
    beta_pars_list[[1]] <- c(runif(1, 0.1, 0.9), runif(1, 0.1, 0.9))
    beta_pars_list[[2]] <- c(runif(1, 0.1, 0.9), runif(1, 1, 10))
    beta_pars_list[[3]] <- c(runif(1, 10, 50), runif(1, 1, 10))
    beta_pars_list[[4]] <- c(runif(1, 5, 15), runif(1, 5, 15))
    beta_pars_list[[5]] <- c(runif(1, 1, 10), runif(1, 5, 15))
    
    rho <- sample(c(-1, 1), 1)*abs_rho
    
    dat_orig <- SimulateCorrelatedBetaData(n = n, 
                                           rho = rho, 
                                           beta_pars_list = beta_pars_list)
    
    dat_hold <- SimulateCorrelatedBetaData(n = n, 
                                           rho = rho, 
                                           beta_pars_list = beta_pars_list)
    
    cat("####################### generate synthetic data", "\n")
    cat("joint factorization", "\n")
    syn_jf <- JointFactorizationTabPFNGenerator(X = dat_orig)
    cat("full conditionals", "\n")
    syn_fc <- FullConditionalsTabPFNGenerator(X = dat_orig)
    cat("miav", "\n")
    syn_miav <- MiavTabPFNGenerator(X = dat_orig)
    cat("smote", "\n")
    syn_smote <- SynthSMOTENC(dat = dat_orig, k = 5)
    
    #######################################
    ## compute Wasserstein distance
    #######################################
    
    aux_type <- GetVariableTypes(dat_orig)
    num_variables <- aux_type$num_variables
    
    wd[i, "hold"] <- WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                         dat2_n = dat_hold[, num_variables])
    
    wd[i, "jf"] <- WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                       dat2_n = syn_jf[, num_variables])    
      
    wd[i, "fc"] <- WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                       dat2_n = syn_fc[, num_variables])
    
    wd[i, "miav"] <- WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                         dat2_n = syn_miav[, num_variables])
    
    wd[i, "smote"] <- WassersteinDistance(dat1_n = dat_orig[, num_variables], 
                                          dat2_n = syn_smote[, num_variables])
    
  }
  
  return(list(wd = wd))
}


#################################################################
## Simulated datasets. 
#################################################################

# To run TabPFN in R: 
# First in a terminal, create a virtual environment, activate it, and install 
# TabPFN. Then download the "generate_tabpfn_predictions.py" script.
#
# In R load reticulate and run the python script, which make classifiers 
# and regression models based on TabPFN available in R.
library(reticulate)
use_virtualenv("~/TabPFN/venv")
source_python("~/TabPFN/venv/generate_tabpfn_predictions.py")


n_sim <- 10
n <- 400

set.seed(123)
out_0.95 <- RunSimulationsWD(n_sim = n_sim, n = n, abs_rho = 0.95, my_seed = 1001)
save(out_0.95, file = "simulation_outputs_wd_abs_rho_0.95.RData", compress = TRUE)

set.seed(123)
out_0.75 <- RunSimulationsWD(n_sim = n_sim, n = n, abs_rho = 0.75, my_seed = 1002)
save(out_0.75, file = "simulation_outputs_wd_abs_rho_0.75.RData", compress = TRUE)

set.seed(123)
out_0.5 <- RunSimulationsWD(n_sim = n_sim, n = n, abs_rho = 0.5, my_seed = 1003)
save(out_0.5, file = "simulation_outputs_wd_abs_rho_0.5.RData", compress = TRUE)

set.seed(123)
out_0.25 <- RunSimulationsWD(n_sim = n_sim, n = n, abs_rho = 0.25, my_seed = 1004)
save(out_0.25, file = "simulation_outputs_wd_abs_rho_0.25.RData", compress = TRUE)

set.seed(123)
out_0 <- RunSimulationsWD(n_sim = n_sim, n = n, abs_rho = 0, my_seed = 1005)
save(out_0, file = "simulation_outputs_wd_abs_rho_0.RData", compress = TRUE)


###########################################################################
## First 21 datasets. 
## Load the outputs generated by the jupyter notebook
## "generate_synthetic_data_for_real_data_experiments_openml_cc18.ipynb"
###########################################################################

library(arrow)

# path to the folder storing the saved outputs
data_path <- ""

# load a large feather dataset containing the original and holdout data splits
# for the first 21 datasets
df_split <- read_feather(paste0(data_path, "openml_cc18_orig_hold_data_splits.feather"))

# load a large feather dataset containing the miav-based synthetic versions
# of each data split of the original data
df_miav <- read_feather(paste0(data_path, "openml_cc18_syn_miav.feather"))

# load a large feather dataset containing the JF-based synthetic versions
# of each data split of the original data
df_jf <- read_feather(paste0(data_path, "openml_cc18_syn_jf.feather"))

# load a large feather dataset containing the FC-based synthetic versions
# of each data split of the original data
df_fc <- read_feather(paste0(data_path, "openml_cc18_syn_fc.feather"))


set.seed(12345)
out_hold <- EvaluateHoldoutDataWD(df_split = df_split)
save(out_hold, file = "real_data_wd_evaluations_21_datasets_hold.RData", compress = TRUE)


set.seed(12345)
out_miav <- EvaluateSyntheticDataWD(df_split = df_split,
                                    df_synth = df_miav)
save(out_miav, file = "real_data_wd_evaluations_21_datasets_miav.RData", compress = TRUE)



set.seed(12345)
out_jf <- EvaluateSyntheticDataWD(df_split = df_split,
                                     df_synth = df_jf)
save(out_jf, file = "real_data_wd_evaluations_21_datasets_jf.RData", compress = TRUE)


set.seed(12345)
out_fc <- EvaluateSyntheticDataWD(df_split = df_split,
                                     df_synth = df_fc)
save(out_fc, file = "real_data_wd_evaluations_21_datasets_fc.RData", compress = TRUE)


set.seed(12345)
out_smote <- EvaluateSmoteDataWD(df_split = df_split, k = 5)
save(out_smote, file = "real_data_wd_evaluations_21_datasets_smote.RData", compress = TRUE)



#######################################################################################
## Additional 15 datasets. 
## Load the outputs generated by the jupyter notebook
## "generate_synthetic_data_for_real_data_experiments_openml_cc18_additional.ipynb"
#######################################################################################

# path to the folder storing the saved outputs
data_path <- ""

# load a large feather dataset containing the original and holdout data splits
# for the additional 15 datasets
df_split_a <- read_feather(paste0(data_path, "openml_cc18_orig_hold_data_splits_additional.feather"))

# load a large feather dataset containing the miav-based synthetic versions
# of each data split of the original data
df_miav_a <- read_feather(paste0(data_path, "openml_cc18_syn_miav_additional.feather"))

# load a large feather dataset containing the JF-based synthetic versions
# of each data split of the original data
df_jf_a <- read_feather(paste0(data_path, "openml_cc18_syn_jf_additional.feather"))

# load a large feather dataset containing the FC-based synthetic versions
# of each data split of the original data
df_fc_a <- read_feather(paste0(data_path, "openml_cc18_syn_fc_additional.feather"))


set.seed(12345)
out_hold_a <- EvaluateHoldoutDataWD(df_split = df_split_a)
save(out_hold_a, file = "real_data_wd_evaluations_15_additional_datasets_hold.RData", compress = TRUE)


set.seed(12345)
out_miav_a <- EvaluateSyntheticDataWD(df_split = df_split_a,
                                    df_synth = df_miav_a)
save(out_miav_a, file = "real_data_wd_evaluations_15_additional_datasets_miav.RData", compress = TRUE)



set.seed(12345)
out_jf_a <- EvaluateSyntheticDataWD(df_split = df_split_a,
                                  df_synth = df_jf_a)
save(out_jf_a, file = "real_data_wd_evaluations_15_additional_datasets_jf.RData", compress = TRUE)


set.seed(12345)
out_fc_a <- EvaluateSyntheticDataWD(df_split = df_split_a,
                                  df_synth = df_fc_a)
save(out_fc_a, file = "real_data_wd_evaluations_15_additional_datasets_fc.RData", compress = TRUE)


set.seed(12345)
out_smote_a <- EvaluateSmoteDataWD(df_split = df_split_a, k = 5)
save(out_smote_a, file = "real_data_wd_evaluations_15_additional_datasets_smote.RData", compress = TRUE)



#########################################################################
## Baseline datasets.
## This script loads the outputs generated by the jupyter notebooks
## "miav_jf_fc_on_baseline_data.ipynb" and
## "synthcity_baseline_comparisons.ipynb"
#########################################################################

# path to the folders storing the saved outputs
data_path <- ""

# list generators names
generator_names <- c("miav", 
                     "jf", 
                     "fc", 
                     "arf", 
                     "ctgan", 
                     "tvae", 
                     "ddpm", 
                     "bayesnet", 
                     "smote")


set.seed(12345)
out_AB <- RunEvaluationsBaselineWD(ds_name = "abalone",
                                   data_path = data_path,
                                   generator_names = generator_names)

set.seed(12345)
out_BM <- RunEvaluationsBaselineWD(ds_name = "bank",
                                   data_path = data_path,
                                   generator_names = generator_names)

set.seed(12345)
out_CR <- RunEvaluationsBaselineWD(ds_name = "credit",
                                   data_path = data_path,
                                   generator_names = generator_names)

set.seed(12345)
out_EM <- RunEvaluationsBaselineWD(ds_name = "eye",
                                   data_path = data_path,
                                   generator_names = generator_names)

set.seed(12345)
out_HO <- RunEvaluationsBaselineWD(ds_name = "house16h",
                                   data_path = data_path,
                                   generator_names = generator_names)

set.seed(12345)
out_MT <- RunEvaluationsBaselineWD(ds_name = "magic",
                                   data_path = data_path,
                                   generator_names = generator_names)

set.seed(12345)
out_PO <- RunEvaluationsBaselineWD(ds_name = "pol",
                                   data_path = data_path,
                                   generator_names = generator_names)



save(out_AB, out_BM, out_CR, out_EM, out_HO, out_MT, out_PO, 
     file = "outputs_wd_real_world_experiments_baseline_comparisons.RData", 
     compress = TRUE)

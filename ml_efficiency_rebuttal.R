
source("utility_functions_for_miav_tabpfn_iclr.R")

##############################################################################
## additional utility functions implementing the ML efficiency evaluations 
##############################################################################

RfMLEfficiency <- function(dat_synt,
                           dat_test,
                           outcome_name, 
                           feature_names,
                           ml_task) {

  ## train of synthetic, evaluate on real
  dat <- rbind(dat_synt, dat_test)
  idx_train <- seq(nrow(dat_synt))
  idx_test <- seq(nrow(dat_synt)+1, nrow(dat))
  out_syn <- switch(ml_task,
                    class = FitRangerClass(dat,
                                           idx_train, 
                                           idx_test, 
                                           outcome_name, 
                                           feature_names),
                    regr = FitRangerRegr(dat,
                                         idx_train, 
                                         idx_test, 
                                         outcome_name, 
                                         feature_names))
  ml_efficiency <- out_syn[[1]]
  
  return(list(ml_efficiency = ml_efficiency,
              out_syn = out_syn))
}


FitRangerClass <- function(dat,
                           idx_train, 
                           idx_test, 
                           outcome_name, 
                           feature_names) {
  ## Inputs:
  ## dat: data.frame containing the features and label data
  ## idx_train: index of the training samples
  ## idx_test: index of the test samples
  ## outcome_name: name of the outcome variable
  ## feature_names: names of the input variables
  ##
  ## Output:
  ## auc_obs:  observed mean AUC score 
  ## auc_list: list of one-vs-all AUC calculations
  
  dat <- dat[, c(outcome_name, feature_names)]
  dat[, outcome_name] <- factor(as.character(dat[, outcome_name])) 
  
  my_formula <- as.formula(paste(outcome_name, " ~ ", 
                                 paste(feature_names, collapse = " + ")))
  fit <- ranger(my_formula, 
                data = dat[idx_train,], 
                probability = TRUE, 
                verbose = FALSE)
  pred_probs <- predict(fit, 
                        dat[idx_test, -1, drop = FALSE], 
                        type = "response")$predictions
  
  auc_list <- list()
  
  outcome_classes <- levels(dat[, outcome_name])
  
  y_test <- dat[idx_test, 1]
  
  for (cls in outcome_classes) {
    # One-vs-rest labels: positive = cls, negative = others
    binary_y_test <- ifelse(y_test == cls, 1, 0)
    
    # Get probabilities for this class
    roc_obj <- roc(binary_y_test, pred_probs[, cls], quiet = TRUE)
    
    auc_list[[cls]] <- pROC::auc(roc_obj)
  }
  
  auc_mean <- mean(unlist(auc_list))
  
  return(list(auc_mean = auc_mean, 
              auc_list = auc_list))
}


FitRangerRegr <- function(dat,
                          idx_train, 
                          idx_test, 
                          outcome_name, 
                          feature_names) {
  ## Inputs:
  ## dat: data.frame containing the features and label data
  ## idx_train: index of the training samples
  ## idx_test: index of the test samples
  ## outcome_name: name of the outcome variable
  ## feature_names: names of the input variables
  ##
  ## Output:
  ## rmse:  root mean squared error
  ## nrmse: normalized root mean squared error
  
  dat <- dat[, c(outcome_name, feature_names)]
  
  my_formula <- as.formula(paste(outcome_name, " ~ ", 
                                 paste(feature_names, collapse = " + ")))
  fit <- ranger(my_formula, 
                data = dat[idx_train,], 
                verbose = FALSE)
  y_hat <- predict(fit, 
                   dat[idx_test, -1, drop = FALSE], 
                   type = "response")$predictions
  
  y_test <- dat[idx_test, 1]
  rmse <- sqrt(mean((y_test - y_hat)^2))
  
  nrmse <- rmse/(max(y_test) - min(y_test))
  
  r2 <- cor(y_test, y_hat)^2
  
  return(list(r2 = r2,
              nrmse = nrmse,
              rmse = rmse))
}


RunSimulationsMLEff <- function(n_sim, n, abs_rho, my_seed) {
  
  sim_seeds <- NULL
  if (!is.null(my_seed)) {
    set.seed(my_seed)
    sim_seeds <- sample(seq(1e+4, 1e+5), n_sim, replace = FALSE)
  }
  
  methods_names <- c("hold", "jf", "fc", "miav", "smote")
  
  mle <- matrix(NA, n_sim, length(methods_names)) 
  colnames(mle) <- methods_names
  
  feat_names <- paste0("X", seq(4))
  target_name <- "X5"
  
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
    ## compute fidelity metrics
    #######################################
    
    
    mle[i, "hold"] <- RfMLEfficiency(dat_synt = dat_orig,
                                     dat_test = dat_hold,
                                     outcome_name = target_name, 
                                     feature_names = feat_names,
                                     ml_task = "regr")[[1]]
    
    mle[i, "jf"] <- RfMLEfficiency(dat_synt = syn_jf,
                                   dat_test = dat_hold,
                                   outcome_name = target_name, 
                                   feature_names = feat_names,
                                   ml_task = "regr")[[1]]
    
    mle[i, "fc"] <- RfMLEfficiency(dat_synt = syn_fc,
                                   dat_test = dat_hold,
                                   outcome_name = target_name, 
                                   feature_names = feat_names,
                                   ml_task = "regr")[[1]]
    
    mle[i, "miav"] <- RfMLEfficiency(dat_synt = syn_miav,
                                     dat_test = dat_hold,
                                     outcome_name = target_name, 
                                     feature_names = feat_names,
                                     ml_task = "regr")[[1]]
    
    mle[i, "smote"] <- RfMLEfficiency(dat_synt = syn_smote,
                                      dat_test = dat_hold,
                                      outcome_name = target_name, 
                                      feature_names = feat_names,
                                      ml_task = "regr")[[1]]
    
  }
  
  return(list(mle = mle))
}


## ML performance of the synthetic data (trains a RF model on the synthetic 
## data, and uses the holdout set as the test set)
EvaluateSyntheticDataMLEff <- function(df_split,
                                       df_synth,
                                       target_names,
                                       ml_tasks) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  mle <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(mle) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)

  ## n_datasets
  for (i in seq(n_datasets)) {
    
    for (j in seq(n_splits)) {
      cat(c(i, j), "\n")
      
      dat_hold <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "hold")
      
      dat_synt <- GrabDataset(df = df_synth, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "syn")
      
      # If data synthesis failed for a dataset (so that it has 0 rows),
      # we skip the evaluation
      if (nrow(dat_synt) > 0) {
        ## add "X" to variable names
        dat_hold <- data.frame(dat_hold)
        dat_synt <- data.frame(dat_synt)
        
        ## create the modified target_name 
        ## (format: X<task_id>__<dataset_name>__<target_name>)
        aux <- names(dat_hold)
        aux <- strsplit(aux, "__")
        aux <- aux[[1]][-3]
        aux <- paste(aux, collapse = "__")
        modified_target_name <- paste(aux, target_names[i], sep = "__")
        
        feat_names <- setdiff(colnames(dat_hold), modified_target_name)
        
        cat("mle ", c(i, j), "\n")
        tmp <- try(RfMLEfficiency(dat_synt = dat_synt,
                                  dat_test = dat_hold,
                                  outcome_name = modified_target_name, 
                                  feature_names = feat_names,
                                  ml_task = ml_tasks[i])[[1]],
                   silent = TRUE)
        if (!inherits(tmp, "try-error")) {
          mle[j, i] <- tmp
        }
      }
    }
  }
  
  return(list(mle = mle))
}


## ML performance on real data (trains a RF model on the training data,
## and uses the holdout set as the test set)
EvaluateGroundTruthMLEff <- function(df_split,
                                       target_names,
                                       ml_tasks) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  mle <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(mle) <- dataset_names
  
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
    
      ## add "X" to variable names
      dat_orig <- data.frame(dat_orig)
      dat_hold <- data.frame(dat_hold)
      
      ## create the modified target_name 
      ## (format: X<task_id>__<dataset_name>__<target_name>)
      aux <- names(dat_hold)
      aux <- strsplit(aux, "__")
      aux <- aux[[1]][-3]
      aux <- paste(aux, collapse = "__")
      modified_target_name <- paste(aux, target_names[i], sep = "__")
      
      feat_names <- setdiff(colnames(dat_hold), modified_target_name)
      
      cat("mle ", c(i, j), "\n")
      mle[j, i] <- RfMLEfficiency(dat_synt = dat_orig,
                                  dat_test = dat_hold,
                                  outcome_name = modified_target_name, 
                                  feature_names = feat_names,
                                  ml_task = ml_tasks[i])[[1]]
    }
  }
  
  return(list(mle = mle))
}


EvaluateSmoteDataMLEff <- function(df_split,
                              k = 5,
                              target_names,
                              ml_tasks) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  mle <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(mle) <- dataset_names

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
      
      ## add "X" to variable names
      dat_orig <- data.frame(dat_orig)
      dat_hold <- data.frame(dat_hold)
      
      ## generate synthetic data using smote
      dat_synt <- SynthSMOTENC(dat_orig, k = k)
      
      
      ## create the modified target_name 
      ## (format: X<task_id>__<dataset_name>__<target_name>)
      aux <- names(dat_hold)
      aux <- strsplit(aux, "__")
      aux <- aux[[1]][-3]
      aux <- paste(aux, collapse = "__")
      modified_target_name <- paste(aux, target_names[i], sep = "__")
      
      feat_names <- setdiff(colnames(dat_hold), modified_target_name)
      
      cat("mle ", c(i, j), "\n")
      tmp <- try(RfMLEfficiency(dat_synt = dat_synt,
                                dat_test = dat_hold,
                                outcome_name = modified_target_name, 
                                feature_names = feat_names,
                                ml_task = ml_tasks[i])[[1]],
                 silent = TRUE)
      if (!inherits(tmp, "try-error")) {
        mle[j, i] <- tmp
      }
    }
    
  }
  
  return(list(mle = mle))
}


EvaluateSyntheticDataBaselineMLEff <- function(df_split,
                                          df_synth,
                                          target_name,
                                          ml_task) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  mle <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(mle) <- dataset_names
  
  for (j in seq(n_splits)) {
    cat(c(j), "\n")
    
    dat_hold <- GrabDataset(df = df_split, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "hold")
    
    dat_synt <- GrabDataset(df = df_synth, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "syn")
  
    dat_hold <- data.frame(dat_hold)
    dat_synt <- data.frame(dat_synt)
    
    feat_names <- setdiff(colnames(dat_hold), target_name)
    
    cat("mle ", j, "\n")
    tmp <- try(RfMLEfficiency(dat_synt = dat_synt,
                              dat_test = dat_hold,
                              outcome_name = target_name, 
                              feature_names = feat_names,
                              ml_task = ml_task)[[1]],
               silent = TRUE)
    if (!inherits(tmp, "try-error")) {
      mle[j, 1] <- tmp
    }
  }
  
  return(list(mle = mle))
}


EvaluateHoldoutDataBaselineMLEff <- function(df_split,
                                             target_name,
                                             ml_task) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  mle <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(mle) <- dataset_names
  
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
    
    feat_names <- setdiff(colnames(dat_hold), target_name)
    
    cat("mle ", j, "\n")
    tmp <- try(RfMLEfficiency(dat_synt = dat_orig,
                              dat_test = dat_hold,
                              outcome_name = target_name, 
                              feature_names = feat_names,
                              ml_task = ml_task)[[1]],
               silent = TRUE)
    if (!inherits(tmp, "try-error")) {
      mle[j, 1] <- tmp
    }
  }
  
  return(list(mle = mle))
}


RunEvaluationsBaselineMLEff <- function(ds_name,
                                        data_path,
                                        generator_names,
                                        target_name,
                                        ml_task) {
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
  aux_list[[1]] <- EvaluateHoldoutDataBaselineMLEff(df_split = df_split, 
                                                    target_name = target_name,
                                                    ml_task = ml_task)
  
  for (i in seq(n_generators)) {
    cat("running evaluations for: ", generator_names[i], "\n")
    aux_list[[i + 1]] <- EvaluateSyntheticDataBaselineMLEff(df_split = df_split,
                                                            df_synth = df_generators[[i]],
                                                            target_name = target_name,
                                                            ml_task = ml_task)
  }
  
  # Organize outputs by evaluation metric
  p <- length(aux_list)
  nms <- names(aux_list)
  
  mle <- aux_list[[1]]$mle
  for (i in seq(2, p)) {
    mle <- cbind(mle, aux_list[[i]]$mle)
  }
  colnames(mle) <- nms
  
  return(list(mle = mle))
}



#################################################################
## simulated datasets 
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
out_0.95 <- RunSimulationsMLEff(n_sim = n_sim, n = n, abs_rho = 0.95, my_seed = 1001)
save(out_0.95, file = "simulation_outputs_mle_abs_rho_0.95.RData", compress = TRUE)

set.seed(123)
out_0.75 <- RunSimulationsMLEff(n_sim = n_sim, n = n, abs_rho = 0.75, my_seed = 1002)
save(out_0.75, file = "simulation_outputs_mle_abs_rho_0.75.RData", compress = TRUE)

set.seed(123)
out_0.5 <- RunSimulationsMLEff(n_sim = n_sim, n = n, abs_rho = 0.5, my_seed = 1003)
save(out_0.5, file = "simulation_outputs_mle_abs_rho_0.5.RData", compress = TRUE)

set.seed(123)
out_0.25 <- RunSimulationsMLEff(n_sim = n_sim, n = n, abs_rho = 0.25, my_seed = 1004)
save(out_0.25, file = "simulation_outputs_mle_abs_rho_0.25.RData", compress = TRUE)

set.seed(123)
out_0 <- RunSimulationsMLEff(n_sim = n_sim, n = n, abs_rho = 0, my_seed = 1005)
save(out_0, file = "simulation_outputs_mle_abs_rho_0.RData", compress = TRUE)



##############################################################################
## First 21 datasets. 
## Load the outputs generated by the jupyter notebook
## "generate_synthetic_data_for_real_data_experiments_openml_cc18.ipynb"
##############################################################################

## names of the target variables for the first 21 datasets
target_names_21 <- c("class", "class", "Class", "class", "class", "class", 
                     "class", "Class", "Author", "c", "c", "problems", 
                     "defects", "Class", "Class", "Class", "Class", "Class", 
                     "class", "target", "outcome")

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
out_miav <- EvaluateSyntheticDataMLEff(df_split = df_split,
                                       df_synth = df_miav,
                                       target_names = target_names_21,
                                       ml_tasks = rep("class", 21))
save(out_miav, file = "real_data_mle_evaluations_21_datasets_miav.RData", compress = TRUE)


set.seed(12345)
out_jf <- EvaluateSyntheticDataMLEff(df_split = df_split,
                                     df_synth = df_jf,
                                     target_names = target_names_21,
                                     ml_tasks = rep("class", 21))
save(out_jf, file = "real_data_mle_evaluations_21_datasets_jf.RData", compress = TRUE)


set.seed(12345)
out_fc <- EvaluateSyntheticDataMLEff(df_split = df_split,
                                     df_synth = df_fc,
                                     target_names = target_names_21,
                                     ml_tasks = rep("class", 21))
save(out_fc, file = "real_data_mle_evaluations_21_datasets_fc.RData", compress = TRUE)


set.seed(12345)
out_hold <- EvaluateGroundTruthMLEff(df_split = df_split,
                                     target_names = target_names_21,
                                     ml_tasks = rep("class", 21))
save(out_hold, file = "real_data_mle_evaluations_21_datasets_hold.RData", compress = TRUE)


set.seed(12345)
out_smote <- EvaluateSmoteDataMLEff(df_split = df_split,
                               k = 5, 
                               target_names = target_names_21, 
                               ml_tasks = rep("class", 21))
save(out_smote, file = "real_data_mle_evaluations_21_datasets_smote.RData", compress = TRUE)




#######################################################################################
## Additional 15 datasets.
## Load the outputs generated by the jupyter notebook
## "generate_synthetic_data_for_real_data_experiments_openml_cc18_additional.ipynb"
#######################################################################################

## names of the target variables for the additional 15 datasets
target_names_15 <- c("class", "class", "class", "class", "defects", "Class", 
                     "Class", "Class", "Class", "Class", "Phase", "class", 
                     "class", "class", "class")

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
out_miav_a <- EvaluateSyntheticDataMLEff(df_split = df_split_a,
                                       df_synth = df_miav_a,
                                       target_names = target_names_15,
                                       ml_tasks = rep("class", 15))
save(out_miav_a, file = "real_data_mle_evaluations_15_additional_datasets_miav.RData", compress = TRUE)


set.seed(12345)
out_jf_a <- EvaluateSyntheticDataMLEff(df_split = df_split_a,
                                     df_synth = df_jf_a,
                                     target_names = target_names_15,
                                     ml_tasks = rep("class", 15))
save(out_jf_a, file = "real_data_mle_evaluations_15_additional_datasets_jf.RData", compress = TRUE)


set.seed(12345)
out_fc_a <- EvaluateSyntheticDataMLEff(df_split = df_split_a,
                                     df_synth = df_fc_a,
                                     target_names = target_names_15,
                                     ml_tasks = rep("class", 15))
save(out_fc_a, file = "real_data_mle_evaluations_15_additional_datasets_fc.RData", compress = TRUE)


set.seed(12345)
out_hold_a <- EvaluateGroundTruthMLEff(df_split = df_split_a,
                                     target_names = target_names_15,
                                     ml_tasks = rep("class", 15))
save(out_hold_a, file = "real_data_mle_evaluations_15_additional_datasets_hold.RData", compress = TRUE)


set.seed(12345)
out_smote_a <- EvaluateSmoteDataMLEff(df_split = df_split_a,
                               k = 5, 
                               target_names = target_names_15, 
                               ml_tasks = rep("class", 15))
save(out_smote_a, file = "real_data_mle_evaluations_15_additional_datasets_smote.RData", compress = TRUE)



########################################################################
## Baseline datasets.
## This script loads the outputs generated by the jupyter notebooks
## "miav_jf_fc_on_baseline_data.ipynb" and
## "synthcity_baseline_comparisons.ipynb"
########################################################################

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


## evaluate all generators w.r.t the fidelity and privacy metrics on the AB data
set.seed(12345)
out_AB <- RunEvaluationsBaselineMLEff(ds_name = "abalone",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "regr")


## evaluate all generators w.r.t the fidelity and privacy metrics on the BM data
set.seed(12345)
out_BM <- RunEvaluationsBaselineMLEff(ds_name = "bank",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "class")


## evaluate all generators w.r.t the fidelity and privacy metrics on the CR data
set.seed(12345)
out_CR <- RunEvaluationsBaselineMLEff(ds_name = "credit",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "class")


## evaluate all generators w.r.t the fidelity and privacy metrics on the EM data
set.seed(12345)
out_EM <- RunEvaluationsBaselineMLEff(ds_name = "eye",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "class")


## evaluate all generators w.r.t the fidelity and privacy metrics on the HO data
set.seed(12345)
out_HO <- RunEvaluationsBaselineMLEff(ds_name = "house16h",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "class")


## evaluate all generators w.r.t the fidelity and privacy metrics on the MT data
set.seed(12345)
out_MT <- RunEvaluationsBaselineMLEff(ds_name = "magic",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "class")


## evaluate all generators w.r.t the fidelity and privacy metrics on the PO data
set.seed(12345)
out_PO <- RunEvaluationsBaselineMLEff(ds_name = "pol",
                                 data_path = data_path,
                                 generator_names = generator_names,
                                 target_name = "target", 
                                 ml_task = "class")



save(out_AB, out_BM, out_CR, out_EM, out_HO, out_MT, out_PO, 
     file = "outputs_mle_real_world_experiments_baseline_comparisons.RData", 
     compress = TRUE)






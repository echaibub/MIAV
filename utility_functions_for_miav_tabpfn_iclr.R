#############################################
## Dependencies
#############################################

library(corrplot)
library(MASS)
library(Rfast)
library(FNN)
library(ranger)
library(pROC)


##############################################
## Utility functions for implementing the
## TabPFN based synthetic data generators
##############################################


GenerateMaximalInformationAuxiliaryVariable <- function(x) {
  m <- sort(runif(length(x)))
  
  if (class(x) %in% c("numeric", "integer")) {
    r <- rank(x, ties.method = "random")
    m <- m[r]
  }
  
  else if (class(x) %in% c("factor", "character", "logical")) {
    x <- as.factor(x)
    r <- NumericRankEncoding(x)
    m <- m[r]
  }
  
  return(m)
}


ComputeAuxiliaryVariables <- function(dat) {
  n <- nrow(dat)
  p <- ncol(dat)
  aux <- data.frame(matrix(NA, n, p))
  colnames(aux) <- paste0("aux", seq(p))
  for (i in seq(p)) {
    aux[, i] <- GenerateMaximalInformationAuxiliaryVariable(dat[, i])
  }
  
  return(aux)
}


NumericRankEncoding <- function(x) {
  tb <- table(x)
  variable_levels <- names(tb)
  cumulative_counts <- cumsum(c(0, as.numeric(tb)))
  n_levels <- length(variable_levels)
  r <- rep(NA, length(x)) ## store the rank encoded numeric variable
  for (j in seq(n_levels)) {
    idx <- which(x == variable_levels[j])
    lower_bound <- cumulative_counts[j] + 1
    upper_bound <- cumulative_counts[j + 1]
    ## shuffle the values ("random tie breaking of ranks")
    r[idx] <- seq(lower_bound, upper_bound)[sample(length(idx))]
  }
  
  return(r)
}



GeneratePredictionUsingTabPFN <- function(X_trn, X_tst, y_trn) {
  target_type <- class(y_trn)
  
  if (target_type %in% c("numeric", "integer")) {
    fit <- generate_tabpfn_regression_prediction(X_train = X_trn, 
                                                 X_test = X_tst, 
                                                 y_train = y_trn)
  }
  if (target_type %in% c("factor", "character", "logical")) {
    fit <- generate_tabpfn_classifier_prediction(X_train = X_trn, 
                                                 X_test = X_tst, 
                                                 y_train = y_trn)
  }
  
  ## extract the prediction
  pred <- fit$predictions$tolist()
  
  return(pred)
}


ICLwithMiavTabPFN <- function(X_trn, 
                              X_tst) {
  p <- ncol(X_tst)
  Z_tst <- data.frame(matrix(NA, nrow(X_tst), p)) ## synthetic data
  colnames(Z_tst) <- colnames(X_tst)
  
  for (j in seq(p)) {
    
    cat("synthesize column ", j, "\n")
    
    m_trn <- GenerateMaximalInformationAuxiliaryVariable(X_trn[, j])
    m_trn <- data.frame(matrix(m_trn, nc = 1, dimnames = list(NULL, "m")))
    
    m_tst <- GenerateMaximalInformationAuxiliaryVariable(X_tst[, j])
    m_tst <- data.frame(matrix(m_tst, nc = 1, dimnames = list(NULL, "m")))
    
    Z_tst[, j] <- GeneratePredictionUsingTabPFN(X_trn = m_trn, 
                                                X_tst = m_tst, 
                                                y_trn = X_trn[, j])
  }
  
  return(Z_tst)
}


MiavTabPFNGenerator <- function(X) {
  
  n <- nrow(X)
  idx1 <- seq(round(n/2))
  X1 <- X[idx1,]
  X2 <- X[-idx1,]
  
  cat("train with X2, query with X1", "\n")
  Z1 <- ICLwithMiavTabPFN(X_trn = X2, X_tst = X1)
  
  cat("train with X1, query with X2", "\n")
  Z2 <- ICLwithMiavTabPFN(X_trn = X1, X_tst = X2)
  
  Z <- rbind(Z1, Z2)
  
  ## TabPFN generates real values for all numeric variables.
  ## This function rounds the real values to the closest integer
  ## for the integer valued variables. 
  Z <- RoundIntegerVariables(X, Z)
  
  return(Z)
}



ICLwithJointFactorizationTabPFN <- function(X_trn, X_tst) {
  n_trn <- nrow(X_trn)
  n_tst <- nrow(X_tst)
  p <- ncol(X_tst)
  Z_tst <- data.frame(matrix(NA, n_tst, p))
  colnames(Z_tst) <- colnames(X_tst)
  
  ## for the first variable
  cat("synthesize column ", 1, "\n")
  X0_tst <- runif(n_tst)
  X0_trn <- runif(n_trn)
  X_trn_less_j <- data.frame(matrix(X0_trn, n_trn, 1))
  X_tst_less_j <- data.frame(matrix(X0_tst, n_tst, 1))
  names(X_trn_less_j) <- names(X_tst_less_j) <- "X0"
  
  Z_tst[, 1] <- GeneratePredictionUsingTabPFN(X_trn = X_trn_less_j, 
                                              X_tst = X_tst_less_j, 
                                              y_trn = X_trn[, 1])
  
  for (j in seq(2, p)) {
    cat("synthesize column ", j, "\n")
    
    X_trn_less_j <- X_trn[, 1:(j-1), drop = FALSE]
    X_tst_less_j <- X_tst[, 1:(j-1), drop = FALSE]
    
    Z_tst[, j] <- GeneratePredictionUsingTabPFN(X_trn = X_trn_less_j, 
                                                X_tst = X_tst_less_j, 
                                                y_trn = X_trn[, j])
  }
  
  return(Z_tst)
}



JointFactorizationTabPFNGenerator <- function(X) {
  n <- nrow(X)
  idx1 <- seq(round(n/2))
  X1 <- X[idx1,]
  X2 <- X[-idx1,]
  
  cat("train with X2, query with X1", "\n")
  Z1 <- ICLwithJointFactorizationTabPFN(X_trn = X2, X_tst = X1)
  
  cat("train with X1, query with X2", "\n")
  Z2 <- ICLwithJointFactorizationTabPFN(X_trn = X1, X_tst = X2)  
  
  Z <- rbind(Z1, Z2)
  
  ## TabPFN generates real values for all numeric variables.
  ## This function rounds the real values to the closest integer
  ## for the integer valued variables. 
  Z <- RoundIntegerVariables(X, Z)
  
  return(Z)
}



ICLwithFullConditionalsTabPFN <- function(X_trn, 
                                          X_tst) {
  p <- ncol(X_tst)
  Z_tst <- data.frame(matrix(NA, nrow(X_tst), p))
  colnames(Z_tst) <- colnames(X_tst)
  
  for (j in seq(p)) {
    cat("synthesize column ", j, "\n")
    
    X_trn_minus_j <- X_trn[, -j, drop = FALSE]
    X_tst_minus_j <- X_tst[, -j, drop = FALSE]
    
    Z_tst[, j] <- GeneratePredictionUsingTabPFN(X_trn = X_trn_minus_j, 
                                                X_tst = X_tst_minus_j, 
                                                y_trn = X_trn[, j])
  }
  
  return(Z_tst)
}



FullConditionalsTabPFNGenerator <- function(X) {
  n <- nrow(X)
  idx1 <- seq(round(n/2))
  X1 <- X[idx1,]
  X2 <- X[-idx1,]
  
  cat("train with X2, query with X1", "\n")
  Z1 <- ICLwithFullConditionalsTabPFN(X_trn = X2, X_tst = X1)
  
  cat("train with X1, query with X2", "\n")
  Z2 <- ICLwithFullConditionalsTabPFN(X_trn = X1, X_tst = X2)  
  
  Z <- rbind(Z1, Z2)
  
  ## TabPFN generates real values for all numeric variables.
  ## This function rounds the real values to the closest integer
  ## for the integer valued variables. 
  Z <- RoundIntegerVariables(X, Z)
  
  return(Z)
}


ICLwithMiavTabPFN_noisy <- function(X_trn, X_tst, percent) {
  n <- nrow(X_tst)
  p <- ncol(X_tst)
  Z_tst <- data.frame(matrix(NA, nrow(X_tst), p)) ## synthetic data
  colnames(Z_tst) <- colnames(X_tst)
  
  for (j in seq(p)) {
    
    cat("synthesize column ", j, "\n")
    
    m_trn <- GenerateMaximalInformationAuxiliaryVariable(X_trn[, j])
    m_trn <- m_trn + rnorm(n, 0, percent * sd(m_trn))
    m_trn <- data.frame(matrix(m_trn, nc = 1, dimnames = list(NULL, "m")))
    
    m_tst <- GenerateMaximalInformationAuxiliaryVariable(X_tst[, j])
    m_tst <- m_tst + rnorm(n, 0, percent * sd(m_tst))
    m_tst <- data.frame(matrix(m_tst, nc = 1, dimnames = list(NULL, "m")))
    
    Z_tst[, j] <- GeneratePredictionUsingTabPFN(X_trn = m_trn, 
                                                X_tst = m_tst, 
                                                y_trn = X_trn[, j])
  }
  
  return(Z_tst)
}


NoisyMiavTabPFNGenerator <- function(X, percent = 0) {
  
  n <- nrow(X)
  idx1 <- seq(round(n/2))
  X1 <- X[idx1,]
  X2 <- X[-idx1,]
  
  cat("train with X2, query with X1", "\n")
  Z1 <- ICLwithMiavTabPFN_noisy(X_trn = X2, X_tst = X1, percent = percent)
  
  cat("train with X1, query with X2", "\n")
  Z2 <- ICLwithMiavTabPFN_noisy(X_trn = X1, X_tst = X2, percent = percent)
  
  Z <- rbind(Z1, Z2)
  
  ## TabPFN generates real values for all numeric variables.
  ## This function rounds the real values to the closest integer
  ## for the integer valued variables. 
  Z <- RoundIntegerVariables(X, Z)
  
  return(Z)
}


##############################################
## Additional utility functions
##############################################


# Take a bootstrap sample of the first variable instead of
# using a random noise variable.
UpdatedJointFactorizationTabPFNGenerator1 <- function(dat) {
  InternalUpdatedJointFactorization1 <- function(dat_ic_trn, dat_ic_tst) {
    n_trn <- nrow(dat_ic_trn)
    n_tst <- nrow(dat_ic_tst)
    p <- ncol(dat_ic_tst)
    dat_syn <- data.frame(matrix(NA, n_tst, p))
    colnames(dat_syn) <- colnames(dat_ic_tst)
    
    ## take a bootstrap sample of the first variable
    dat_syn[, 1] <- dat_ic_tst[sample(seq(n_tst), n_tst, replace = TRUE), 1]
    
    for (j in seq(2, p)) {
      cat("synthesize column ", j, "\n")
      
      # Use the predictions generated in the previous steps.
      X_ic_tst_less_j <- dat_syn[, 1:(j-1), drop = FALSE]
      X_ic_trn_less_j <- dat_ic_trn[, 1:(j-1), drop = FALSE]
      y_ic_trn <- dat_ic_trn[, j]
      
      ## extract the prediction
      dat_syn[, j] <- GeneratePredictionUsingTabPFN(X_trn = X_ic_trn_less_j, 
                                                    X_tst = X_ic_tst_less_j, 
                                                    y_trn = y_ic_trn)
    }
    
    return(list(dat_syn = dat_syn))
  }
  
  n <- nrow(dat)
  idx1 <- seq(round(n/2))
  dat1 <- dat[idx1,]
  dat2 <- dat[-idx1,]
  ## train on dat1, evaluate on dat2
  cat("train on dat1, evaluate on dat2", "\n")
  aux12 <- InternalUpdatedJointFactorization1(dat_ic_trn = dat1, 
                                              dat_ic_tst = dat2)
  ## train on dat2, evaluate on dat1
  cat("train on dat2, evaluate on dat1", "\n")
  aux21 <- InternalUpdatedJointFactorization1(dat_ic_trn = dat2, 
                                              dat_ic_tst = dat1)
  dat_syn <- rbind(aux12$dat_syn, aux21$dat_syn)
  
  ## TabPFN generates real values for all numeric variables.
  ## This function rounds the real values to the closest integer
  ## for the integer valued variables. 
  dat_syn <- RoundIntegerVariables(dat, dat_syn)
  
  return(dat_syn)
}


UpdatedJointFactorizationTabPFNGenerator2 <- function(dat) {
  InternalUpdatedJointFactorization2 <- function(dat_ic_trn, dat_ic_tst) {
    n_trn <- nrow(dat_ic_trn)
    n_tst <- nrow(dat_ic_tst)
    p <- ncol(dat_ic_tst)
    dat_syn <- data.frame(matrix(NA, n_tst, p))
    colnames(dat_syn) <- colnames(dat_ic_tst)
    
    ## for the first variable
    cat("synthesize column ", 1, "\n")
    X0_ic_tst <- runif(n_tst)
    X0_ic_trn <- runif(n_trn)
    X_ic_tst_less_j <- data.frame(matrix(X0_ic_tst, n_tst, 1))
    X_ic_trn_less_j <- data.frame(matrix(X0_ic_trn, n_trn, 1))
    names(X_ic_tst_less_j) <- names(X_ic_trn_less_j) <- "X0"
    
    y_ic_trn <- dat_ic_trn[, 1]
    ## extract the prediction
    dat_syn[, 1] <- GeneratePredictionUsingTabPFN(X_trn = X_ic_trn_less_j, 
                                                  X_tst = X_ic_tst_less_j, 
                                                  y_trn = y_ic_trn)
    
    
    for (j in seq(2, p)) {
      cat("synthesize column ", j, "\n")
      
      # Use the predictions generated in the previous steps.
      X_ic_tst_less_j <- dat_syn[, 1:(j-1), drop = FALSE]
      X_ic_trn_less_j <- dat_ic_trn[, 1:(j-1), drop = FALSE]
      y_ic_trn <- dat_ic_trn[, j]

      ## extract the prediction
      dat_syn[, j] <- GeneratePredictionUsingTabPFN(X_trn = X_ic_trn_less_j, 
                                                    X_tst = X_ic_tst_less_j, 
                                                    y_trn = y_ic_trn)
    }
    
    return(list(dat_syn = dat_syn, X0_ic_tst = X0_ic_tst))
  }
  
  n <- nrow(dat)
  idx1 <- seq(round(n/2))
  dat1 <- dat[idx1,]
  dat2 <- dat[-idx1,]
  ## train on dat1, evaluate on dat2
  cat("train on dat1, evaluate on dat2", "\n")
  aux12 <- InternalUpdatedJointFactorization2(dat_ic_trn = dat1, 
                                             dat_ic_tst = dat2)
  ## train on dat2, evaluate on dat1
  cat("train on dat2, evaluate on dat1", "\n")
  aux21 <- InternalUpdatedJointFactorization2(dat_ic_trn = dat2, 
                                             dat_ic_tst = dat1)
  dat_syn <- rbind(aux12$dat_syn, aux21$dat_syn)
  
  ## TabPFN generates real values for all numeric variables.
  ## This function rounds the real values to the closest integer
  ## for the integer valued variables. 
  dat_syn <- RoundIntegerVariables(dat, dat_syn)
  
  return(dat_syn)
}



CategorizeVariable <- function(x, n_levels) {
  ## Inputs:
  ## x: vector of numeric values
  ## n_levels: number of categories/levels (i.e., n_c in the paper's notation)
  ##
  ## Output:
  ## out: vector of categorical variables
  
  breaks <- unique(quantile(x, probs = seq(0, 1, by = 1/n_levels)))
  var_levels <- seq(length(breaks)-1)
  out <- cut(x, breaks = breaks, labels = var_levels, include.lowest = TRUE)
  out <- as.character(as.numeric(out))
  
  return(out)
}


CategorizeVariable2 <- function(x, n_levels) {
  var_levels <- seq(n_levels)
  out <- cut(x, breaks = n_levels, labels = var_levels)
  out <- as.character(as.numeric(out))
  
  return(out)
}



SynthSMOTENC <- function(dat, k, round_integer_variables = TRUE) {
  
  ## Implements SMOTE (at this point it handles only numerical variables,
  ## categorical variables are returned without any changes)
  ##
  ## Parameters:
  ## dat: R dataframe containing the data
  ## k: parameter for nearest neighbors
  ##
  ## Returns:
  ## out: data.frame containing the synthetic data
  
  GenerateSynthNumData <- function(dat, 
                                   knn_output, 
                                   num_variables) {
    n <- nrow(dat)
    synthetic_dat <- dat
    for (i in seq(n)) {
      ## Get a sample
      x <- as.numeric(dat[i, num_variables])
      ## Get the sample's k-nearest neighbors
      nn_idx <- knn_output$nn.index[i, ]
      ## Randomly select one of the k-nearest neighbors
      nn_idx <- sample(nn_idx, 1)
      x_nn <- as.numeric(dat[nn_idx, num_variables])
      ## Generate synthetic sample using interpolation
      lambda <- runif(1)
      synthetic_dat[i, num_variables] <- x + lambda * (x_nn - x)
    }
    return(synthetic_dat)
  }
  
  GenerateSynthCatData <- function(dat, 
                                   knn_output,
                                   cat_variables) {
    n <- nrow(dat)
    synthetic_dat <- dat
    for (i in seq(n)) {
      ## Get the k-nearest neighbors from the i-th sample
      nn_idx <- knn_output$nn.index[i, ]
      ## Generate synthetic sample using majority vote (sample + neighbors)
      synthetic_dat[i, cat_variables] <- sapply(cat_variables, function(col) {
        values <- c(as.character(dat[i, col]), as.character(dat[nn_idx, col]))
        levels <- names(sort(table(values), decreasing = TRUE))
        return(levels[1])
      })
    }
    return(synthetic_dat)
  }
  
  ## determine variable types
  aux_type <- GetVariableTypes(dat)
  num_variables <- aux_type$num_variables
  cat_variables <- aux_type$cat_variables
  
  ## Find k-nearest neighbors
  knn_out <- get.knn(dat[, num_variables], k = k)
  
  ## Generate synthetic data for the numeric variables alone
  ## (the categorical data is unchanged)
  synth_num_dat <- GenerateSynthNumData(dat = dat, 
                                        knn_output = knn_out, 
                                        num_variables = num_variables)
  colnames(synth_num_dat) <- colnames(dat)
  
  ## Generate synthetic data for the categorical variables alone
  ## (the numeric data is unchanged)
  synth_cat_dat <- GenerateSynthCatData(dat = dat, 
                                        knn_output = knn_out, 
                                        cat_variables = cat_variables)
  colnames(synth_cat_dat) <- colnames(dat)  
  
  ## combine the synthetic datasets
  synthetic_dat <- synth_num_dat
  synthetic_dat[, cat_variables] <- synth_cat_dat[, cat_variables]
  
  if (round_integer_variables) {
    ## SMOTE generates real values for all numeric variables.
    ## This function rounds the real values to the closest integer
    ## for the integer valued variables. 
    synthetic_dat <- RoundIntegerVariables(dat, synthetic_dat)
  }
  
  return(synthetic_dat)
}




GetVariableTypes <- function(df) {
  # Get the classes of all columns
  classes <- sapply(df, class, simplify = FALSE)
  classes <- lapply(classes, function(x) x[1])
  
  # Treat integer and numeric as numeric
  numeric_vars <- names(classes)[classes %in% c("numeric", "integer")]
  
  # Treat factor, character, logical as categorical
  categorical_vars <- names(classes)[classes %in% c("ordered", "factor", "character", "logical")]
  
  var_names <- colnames(df)
  num_variables <- which(var_names %in% numeric_vars) ## numeric variables column indexes
  cat_variables <- which(var_names %in% categorical_vars) ## categorical variables column indexes
  
  if (length(num_variables) == 0) {
    num_variables <- NULL
  }
  if (length(cat_variables) == 0) {
    cat_variables <- NULL
  }
  
  # Return as a list
  return(list(num_variables = num_variables,
              cat_variables = cat_variables,
              num_variable_names = numeric_vars,
              cat_variable_names = categorical_vars))
}


RoundIntegerVariables <- function(df_ori,
                                  df_syn) {
  ## This function uses the original data to determine which variables
  ## have integer type and then round the values of the corresponding
  ## variables in the synthetic data to the nearest integer value
  ## Inputs:
  ##   df_ori: dataframe containing the original data
  ##   df_syn: dataframe containing the synthetic data
  ## Outputs:
  ##   df_syn: synthetic data with rounded values for the integer 
  ##            valued variables
  
  aux_type <- GetVariableTypes(df_ori)
  num_variables <- aux_type$num_variables
  
  if (!is.null(num_variables)) {
    for (i in num_variables) {
      ## test if numeric column i is of integer type
      if (all.equal(df_ori[, i], round(df_ori[, i])) == TRUE) {
        ## round synthetic data variable to closest integer value
        df_syn[, i] <- round(df_syn[, i]) 
      }
    }
  }
  
  return(df_syn)
}



SimulateCorrelatedBetaData <- function(n, rho, beta_pars_list) {
  GaussianCopula <- function(n, Sigma) {
    p <- nrow(Sigma)
    L <- t(chol(Sigma))
    Z <- matrix(rnorm(p * n), p, n)
    X <- L %*% Z
    U <- t(apply(X, 2, pnorm))
    return(U)
  }
  CreateCorrelationMatrix <- function(rho, p) {
    aux1 <- matrix(rep(1:p, p), p, p)
    aux2 <- matrix(rep(1:p, each = p), p, p) 
    return(rho^abs(aux1 - aux2))
  }
  p <- length(beta_pars_list)
  Sigma <- CreateCorrelationMatrix(rho, p)
  U <- GaussianCopula(n, Sigma)
  X <- matrix(NA, n, p)
  for (j in seq(p)) {
    beta_pars <- beta_pars_list[[j]]
    X[, j] <- qbeta(U[, j], beta_pars[1], beta_pars[2])
  }
  colnames(X) <- paste0("X", seq(p))
  X <- data.frame(X)
  
  return(X)
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



NumericEdist <- function(dat1_n, dat2_n, alpha = 0) {
  dat1_n <- apply(dat1_n, 2, MyScaling, alpha)
  dat2_n <- apply(dat2_n, 2, MyScaling, alpha)
  n1 <- nrow(dat1_n)
  n2 <- nrow(dat2_n)
  D11 <- Rfast::dista(dat1_n, dat1_n)
  D22 <- Rfast::dista(dat2_n, dat2_n)
  D12 <- Rfast::dista(dat1_n, dat2_n)
  m11 <- sum(D11)/(n1*n1)
  m22 <- sum(D22)/(n2*n2)
  m12 <- sum(D12)/(n1*n2)
  ed <- 2 * m12 - m11 - m22  
  
  return(list(ed = ed,
              m11 = m11,
              m22 = m22,
              m12 = m12))
}


ComputeScaledDCR <- function(dat_o, 
                             dat_s, 
                             cat_variables, 
                             distance_type = "euclidean",
                             alpha = 0) {
  
  ## compute the distance matrix
  ##
  ## we set: xnew = dat_s
  ##         x = dat_o
  ##
  ## first row of dm:
  ## dm[1, 1] = distance between dat_s[1,] and dat_o[1,]
  ## dm[1, 2] = distance between dat_s[1,] and dat_o[2,]
  ## dm[1, 3] = distance between dat_s[1,] and dat_o[3,]
  ## ...
  ##
  ## second row of dm:
  ## dm[2, 1] = distance between dat_s[2,] and dat_o[1,]
  ## dm[2, 2] = distance between dat_s[2,] and dat_o[2,]
  ## dm[2, 3] = distance between dat_s[2,] and dat_o[3,]
  ## ...
  if (length(cat_variables) > 0) {
    dat_o <- apply(dat_o[, -cat_variables], 2, MyScaling, alpha)
    dat_s <- apply(dat_s[, -cat_variables], 2, MyScaling, alpha)
  }
  else {
    dat_o <- apply(dat_o, 2, MyScaling, alpha)
    dat_s <- apply(dat_s, 2, MyScaling, alpha)
  }
  
  dm <- dista(xnew = as.matrix(dat_s), 
              x = as.matrix(dat_o), 
              type = distance_type)
  
  ## compute the minimal distance vector
  ## dm[1] = minimal distance between the first row of 
  ##         dat_s and all rows of dat_o
  ## dm[2] = minimal distance between the second row of 
  ##         dat_s and all rows of dat_o
  ## ...
  dm <- apply(dm, 1, min)
  dm <- as.vector(dm)
  
  return(dm)
}



AverageKLDivergenceCat <- function(dat_o,
                                   dat_s,
                                   smoothing = 1e-12,
                                   symmetric = TRUE) {
  kl_divergence_categorical <- function(P, Q, base = NULL, smoothing = 1e-12, dropna = TRUE) {
    stopifnot(smoothing >= 0)
    # helpers
    is_named_numeric <- function(x) is.numeric(x) && !is.null(names(x))
    sample_to_counts <- function(x, dropna) {
      if (dropna) {
        x <- x[!is.na(x)]
        tbl <- table(x, useNA = "no")
      } else {
        tbl <- table(x, useNA = "ifany")  # NA bucket named "<NA>"
      }
      v <- as.numeric(tbl)
      names(v) <- names(tbl)
      v
    }
    normalize_probs <- function(counts, k) {
      (counts + smoothing) / (sum(counts) + smoothing * k)
    }
    # support (union of categories) -------------------------------------
    countsP <- if (is_named_numeric(P)) P else sample_to_counts(P, dropna = dropna)
    countsQ <- if (is_named_numeric(Q)) Q else sample_to_counts(Q, dropna = dropna)
    cats <- union(names(countsP), names(countsQ))
    # align counts to full support
    get_counts_on <- function(cnts, cats) {
      out <- numeric(length(cats))
      names(out) <- cats
      out[names(cnts)] <- cnts
      out
    }
    cP <- get_counts_on(countsP, cats)
    cQ <- get_counts_on(countsQ, cats)
    # probabilities with additive smoothing
    k <- length(cats)
    p <- normalize_probs(cP, k)
    q <- normalize_probs(cQ, k)
    # KL = sum p_i * log(p_i/q_i), skip p_i == 0 components
    mask <- p > 0
    ratio <- p[mask] / q[mask]
    # guard tiny numerical issues
    ratio[ratio <= 0] <- .Machine$double.xmin
    log_term <- if (is.null(base)) log(ratio) else log(ratio, base = base)
    sum(p[mask] * log_term)
  }
  sym_kl_categorical <- function(P, Q, base = NULL, smoothing = 1e-12, dropna = TRUE) {
    0.5 * (
      kl_divergence_categorical(P, Q, base = base, smoothing = smoothing, dropna = dropna) +
        kl_divergence_categorical(Q, P, base = base, smoothing = smoothing, dropna = dropna)
    )
  }
  p <- ncol(dat_o)
  divs <- rep(NA, p)
  for (i in seq(p)) {
    divs[i] <- sym_kl_categorical(dat_o[, i], dat_s[i])
  }
  
  return(ave_kl_diver = mean(divs, na.rm = TRUE))
}



CramerV <- function(v1, v2) {
  n <- length(v1)
  n1 <- length(unique(v1))
  n2 <- length(unique(v2))
  chisq.stat <- as.numeric(chisq.test(v1, v2, correct = TRUE)$statistic)
  
  return(sqrt(chisq.stat/(n * min(c(n1-1, n2-1)))))
}



## Computes the square root of the R2 of a linear model where
## the response is a numeric variable and the covariate is a 
## categorical variable. (This reduces to the absolute value 
## of the correlation when the covariate is numeric.)
NumCatCor <- function(dat,
                      num_var,
                      cat_var) {
  aux <- summary(lm(dat[, num_var] ~ dat[, cat_var]))
  
  return(sqrt(aux$r.squared))
}



ComputeAssociationMatrix <- function(dat,
                                     num_variables,
                                     cat_variables) {
  ## get variable names
  nms <- colnames(dat)
  
  n_num <- length(num_variables)
  n_cat <- length(cat_variables)
  
  AM <- matrix(NA, n_num + n_cat, n_num + n_cat)
  diag(AM) <- 1
  rownames(AM) <- nms
  colnames(AM) <- nms
  
  ## compute pearson correlations between numeric variables
  if (n_num > 1) {
    for (i in num_variables) {
      for (j in num_variables) {
        AM[i, j] <- cor(dat[, i], dat[, j])
        AM[j, i] <- AM[i, j]
      }
    }
  }
  
  ## compute Cramer V statistics between categorical variables
  if (n_cat > 1) {
    for (i in cat_variables) {
      for (j in cat_variables) {
        # This can break for synthetic datasets with variables 
        # with a single level
        cvstat <- try(CramerV(dat[, i], dat[, j]), silent = TRUE)
        if (!inherits(cvstat, "try-error")) {
          AM[i, j] <- cvstat
          AM[j, i] <- AM[i, j]
        }
      }
    }
  }  
  
  ## compute sqrt(R2) between numeric and categorical variables
  if (n_num > 0 & n_cat > 0) {
    for (i in num_variables) {
      for (j in cat_variables) {
        AM[i, j] <- NumCatCor(dat = dat, num_var = i, cat_var = j)
        AM[j, i] <- AM[i, j]
      }
    }
  }
  
  return(AM)
}



L2DistAssociationMatrix <- function(am1, am2) {
  aux <- (abs(am1 - am2))^2
  aux[lower.tri(aux)] <- NA
  diag(aux) <- NA
  
  return(mean(aux, na.rm = TRUE))
}


L2DistCorMatrix <- function(cor1, cor2) {
  aux <- (abs(cor1 - cor2))^2
  aux[lower.tri(aux)] <- NA
  diag(aux) <- NA
  
  return(mean(aux, na.rm = TRUE))
}



DBRL <- function(dat_o, dat_m) {
  IndexesOfMinimum <- function(x) {
    x <- as.numeric(x)
    min_x <- min(x)
    return(which(x == min_x))
  }
  n <- nrow(dat_m)
  dm <- dista(xnew = as.matrix(dat_m), 
              x = as.matrix(dat_o), 
              type = "euclidean")
  idx_min <- apply(dm, 1, IndexesOfMinimum, simplify = FALSE)
  idx_seq <- seq(n)
  flag <- sapply(idx_seq, function(i) ifelse(i %in% idx_min[[i]], 1, 0))
  disclosure_risk <- sum(flag)/n
  
  return(disclosure_risk)
}


DistanceBasedRecordLinkage <- function(dat_o, 
                                       dat_m, 
                                       num_variables,
                                       sort_data = FALSE) {
  dat_o <- dat_o[, num_variables]
  dat_m <- dat_m[, num_variables]
  dat_o <- scale(dat_o)
  dat_m <- scale(dat_m)
  p <- ncol(dat_o)
  if (sort_data) {
    disclosure_risks <- rep(NA, p)
    for (j in seq(p)) {
      idx_o <- order(dat_o[, j])
      idx_m <- order(dat_m[, j])
      dat_o_sorted <- dat_o[idx_o,]
      dat_m_sorted <- dat_m[idx_m,]
      ## compute DBRL on sorted data
      disclosure_risks[j] <- DBRL(dat_o_sorted, dat_m_sorted)
    }
    disclosure_risk <- max(disclosure_risks, na.rm = TRUE) ## get worst case
  }
  else {
    disclosure_risk <- DBRL(dat_o, dat_m)
  }
  
  return(disclosure_risk)
}




## computes the SSDID metric
StandardDeviationIntervalDistance <- function(dat_o, 
                                              dat_m, 
                                              num_variables,
                                              k,
                                              sort_data = FALSE) {
  SDID <- function(dat_o, dat_m, k) {
    n <- nrow(dat_o)
    p <- ncol(dat_o)
    # Compute standard deviation for each column
    sds <- apply(dat_m, 2, sd)
    # Expand to n x p matrix of interval lengths
    interval_lengths <- matrix(rep(sds * k, each = n), nrow = n, byrow = FALSE)
    # Compute lower and upper bounds
    lower_bounds <- dat_m - interval_lengths
    upper_bounds <- dat_m + interval_lengths
    # Vectorized indicator matrix: 1 if dat_o in [lower, upper]
    ind <- (dat_o >= lower_bounds) & (dat_o <= upper_bounds)
    # For each row, count how many columns match
    record_ind <- rowSums(ind)
    # Disclosure risk: fraction of rows where all p columns match
    disclosure_risk <- sum(record_ind == p) / n
    return(disclosure_risk)
  }
  dat_o <- dat_o[, num_variables]
  dat_m <- dat_m[, num_variables]
  if (sort_data) {
    p <- ncol(dat_o)
    disclosure_risks <- rep(NA, p)
    for (j in seq(p)) {
      idx_o <- order(dat_o[, j])
      idx_m <- order(dat_m[, j])
      dat_o_sorted <- dat_o[idx_o,]
      dat_m_sorted <- dat_m[idx_m,]
      ## compute DBRL on sorted data
      disclosure_risks[j] <- SDID(dat_o_sorted, dat_m_sorted, k)
    }
    disclosure_risk <- max(disclosure_risks) ## get worst case
  }
  else {
    disclosure_risk <- SDID(dat_o, dat_m, k)
  }
  
  return(disclosure_risk)
}


## computes and averages the SDID over the k parameter grid
AverageSDID <- function(dat_o, 
                        dat_m, 
                        num_variables,
                        k_grid = seq(0.01, 0.10, by = 0.01),
                        sort_data = FALSE) {
  n_k <- length(k_grid)
  aux <- rep(NA, n_k)
  for (i in seq(n_k)) {
    aux[i] <- StandardDeviationIntervalDistance(dat_o, 
                                                dat_m,
                                                num_variables,
                                                k_grid[i],
                                                sort_data)
  }
  disclosure_risk <- mean(aux)
  
  return(disclosure_risk)
}


RfDetectionTest <- function(dat_o,
                            dat_m,
                            n_runs,
                            feature_names,
                            verbose = TRUE) {
  FitRangerClass <- function(dat_train,
                             dat_test, 
                             label_name, 
                             feature_names,
                             neg_class_name, 
                             pos_class_name) {
    ## Inputs:
    ## dat: data.frame containing the features and label data
    ## idx_train: index of the training samples
    ## idx_test: index of the test samples
    ## label_name: name of the outcome variable
    ## feature_names: names of the input variables
    ## neg_class_name: label level for the negative examples
    ## pos_class_name: label level for the positive examples
    ##
    ## Output:
    ## auc_obs: observed AUC score 
    ## pred_probs: vector with the predicted probabilities of the test set
    ##             examples being a positive case 
    ## roc_obj: roc object fit
    
    dat_train <- dat_train[, c(label_name, feature_names)]
    dat_train[, label_name] <- factor(as.character(dat_train[, label_name]), 
                                      levels = c(neg_class_name, pos_class_name)) 
    
    dat_test <- dat_test[, c(label_name, feature_names)]
    dat_test[, label_name] <- factor(as.character(dat_test[, label_name]), 
                                     levels = c(neg_class_name, pos_class_name))     
    
    my_formula <- as.formula(paste(label_name, " ~ ", 
                                   paste(feature_names, collapse = " + ")))
    fit <- ranger(my_formula, 
                  data = dat_train, 
                  probability = TRUE, 
                  verbose = FALSE)
    pred_probs <- predict(fit, 
                          dat_test[, -1, drop = FALSE], 
                          type = "response")$predictions
    y_test <- dat_test[, 1]
    roc_obj <- roc(y_test,
                   pred_probs[, pos_class_name], quiet = TRUE)    
    auc_obs <- pROC::auc(roc_obj)[1]
    
    list(auc_obs = auc_obs, 
         pred_probs = pred_probs[, pos_class_name], 
         roc_obj = roc_obj)
  }
  
  aucs <- rep(NA, n_runs)
  
  dat_o <- data.frame(dat_o)
  dat_m <- data.frame(dat_m)
  
  n <- nrow(dat_o)
  dat_o$data <- rep("real", n)
  dat_m$data <- rep("synth", n)
  
  ## Order the datasets according to the first numeric variable.
  ## This is done to avoid having repeats of the training data 
  ## in the test data. (E.g., when dat_m is a simple random shuffle
  ## of the rows of dat_o.)
  aux_type <- GetVariableTypes(dat_o)
  num_variables <- aux_type$num_variables
  
  ## get number of distinct values of each numeric variable
  aux_uv <- apply(dat_o[, num_variables], 2, function(x) length(unique(x)))
  
  ## get index of variable with most unique values
  idx_u <- which.max(aux_uv)
  
  idx <- order(dat_o[, num_variables[idx_u]])
  dat_o <- dat_o[idx,]
  
  idx <- order(dat_m[, num_variables[idx_u]])
  dat_m <- dat_m[idx,]
  
  seq_n <- seq(n)
  
  for (i in seq(n_runs)) {
    if (verbose) {
      cat(i, "\n")
    }
    
    idx_train <- sample(n, ceiling(n/2), replace = FALSE)
    idx_test <- setdiff(seq_n, idx_train)
    
    ## This split is done to avoid having repeats of the training data 
    ## in the test data. (E.g., when dat_m is a simple random shuffle
    ## of the rows of dat_o.)
    dat_train <- rbind(dat_o[idx_train,], dat_m[idx_train,]) 
    dat_test <- rbind(dat_o[idx_test,], dat_m[idx_test,])
    
    dat_train <- dat_train[sample(nrow(dat_train)),]
    dat_test <- dat_test[sample(nrow(dat_test)),]
    
    rf <- FitRangerClass(dat_train = dat_train,
                         dat_test = dat_test, 
                         label_name = "data", 
                         feature_names = feature_names,
                         neg_class_name = "real", 
                         pos_class_name = "synth")
    
    aucs[i] <- rf$auc_obs
  }
  
  return(list(median_auc = median(aucs),
              aucs = aucs))
}


AverageKSTestStat <- function(dat_o, dat_s) {
  p <- ncol(dat_o)
  ks_stat <- rep(NA, p)
  for (i in seq(p)) {
    ks_stat[i] <- ks.test(dat_o[, i], dat_s[, i])$statistic
  }
  ave_ks_stat <- mean(ks_stat)
  
  return(ave_ks_stat)
}



######################################
## Experiment's functions
######################################


RunSimulations <- function(n_sim, n, abs_rho, my_seed, percent_grid) {
  
  sim_seeds <- NULL
  if (!is.null(my_seed)) {
    set.seed(my_seed)
    sim_seeds <- sample(seq(1e+4, 1e+5), n_sim, replace = FALSE)
  }
  
  n_noisy <- length(percent_grid)
  
  methods_names <- c("hold", "jf", "fc", "miav", "smote", paste0("miav_", percent_grid))
  
  ks_test_stat <- matrix(NA, n_sim, length(methods_names)) 
  colnames(ks_test_stat) <- methods_names
  
  l2corrdist <- matrix(NA, n_sim, length(methods_names)) 
  colnames(l2corrdist) <- methods_names
  
  ed <- matrix(NA, n_sim, length(methods_names)) 
  colnames(ed) <- methods_names
  
  dt <- matrix(NA, n_sim, length(methods_names)) 
  colnames(dt) <- methods_names  
  
  median_dcrs <- matrix(NA, n_sim, length(methods_names)) 
  colnames(median_dcrs) <- methods_names
  
  dbrls <- matrix(NA, n_sim, length(methods_names)) 
  colnames(dbrls) <- methods_names
  
  sdids <- matrix(NA, n_sim, length(methods_names)) 
  colnames(sdids) <- methods_names
  
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
    
    syn_noisy_miav <- vector(mode = "list", length = n_noisy)
    for (j in seq(n_noisy)) {
      cat("miav percent ", percent_grid[j], "\n")
      syn_noisy_miav[[j]] <- NoisyMiavTabPFNGenerator(X = dat_orig, 
                                                      percent = percent_grid[j])
    }
    
    
    #######################################
    ## compute fidelity metrics
    #######################################
    
    cat("####################### compute fidelity metrics", "\n")
    
    ks_test_stat[i, "hold"] <- AverageKSTestStat(dat_o = dat_orig, 
                                                 dat_s = dat_hold)
    
    ks_test_stat[i, "jf"] <- AverageKSTestStat(dat_o = dat_orig, 
                                               dat_s = syn_jf) 
    
    ks_test_stat[i, "fc"] <- AverageKSTestStat(dat_o = dat_orig, 
                                               dat_s = syn_fc)
    
    ks_test_stat[i, "miav"] <- AverageKSTestStat(dat_o = dat_orig, 
                                                 dat_s = syn_miav)
    
    ks_test_stat[i, "smote"] <- AverageKSTestStat(dat_o = dat_orig, 
                                                  dat_s = syn_smote)
    
    for (j in seq(n_noisy)) {
      ks_test_stat[i, j + 5] <- AverageKSTestStat(dat_o = dat_orig, 
                                                  dat_s = syn_noisy_miav[[j]])
    }
    
    
    cor_orig <- cor(dat_orig)
    
    l2corrdist[i, "hold"] <- L2DistCorMatrix(cor_orig, cor(dat_hold))
    
    l2corrdist[i, "jf"] <- L2DistCorMatrix(cor_orig, cor(syn_jf))
    
    l2corrdist[i, "fc"] <- L2DistCorMatrix(cor_orig, cor(syn_fc))
    
    l2corrdist[i, "miav"] <- L2DistCorMatrix(cor_orig, cor(syn_miav))
    
    l2corrdist[i, "smote"] <- L2DistCorMatrix(cor_orig, cor(syn_smote))
    
    for (j in seq(n_noisy)) {
      l2corrdist[i, j + 5] <- L2DistCorMatrix(cor_orig, cor(syn_noisy_miav[[j]]))
    }
    
    
    ed[i, "hold"] <- NumericEdist(dat1_n = dat_orig, dat2_n = dat_hold)$ed
    
    ed[i, "jf"] <- NumericEdist(dat1_n = dat_orig, dat2_n = syn_jf)$ed
    
    ed[i, "fc"] <- NumericEdist(dat1_n = dat_orig, dat2_n = syn_fc)$ed
    
    ed[i, "miav"] <- NumericEdist(dat1_n = dat_orig, dat2_n = syn_miav)$ed
    
    ed[i, "smote"] <- NumericEdist(dat1_n = dat_orig, dat2_n = syn_smote)$ed
    
    for (j in seq(n_noisy)) {
      ed[i, j + 5] <- NumericEdist(dat1_n = dat_orig, dat2_n = syn_noisy_miav[[j]])$ed
    }
    
    dt[i, "hold"] <- RfDetectionTest(dat_o = dat_orig, 
                                     dat_m = dat_hold, 
                                     n_runs = 5, 
                                     feature_names = colnames(dat_orig),
                                     verbose = FALSE)$median_auc
    
    dt[i, "jf"] <- RfDetectionTest(dat_o = dat_orig, 
                                   dat_m = syn_jf, 
                                   n_runs = 5, 
                                   feature_names = colnames(dat_orig),
                                   verbose = FALSE)$median_auc
    
    dt[i, "fc"] <- RfDetectionTest(dat_o = dat_orig, 
                                   dat_m = syn_fc, 
                                   n_runs = 5, 
                                   feature_names = colnames(dat_orig),
                                   verbose = FALSE)$median_auc
    
    dt[i, "miav"] <- RfDetectionTest(dat_o = dat_orig, 
                                     dat_m = syn_miav, 
                                     n_runs = 5, 
                                     feature_names = colnames(dat_orig),
                                     verbose = FALSE)$median_auc
    
    dt[i, "smote"] <- RfDetectionTest(dat_o = dat_orig, 
                                      dat_m = syn_smote, 
                                      n_runs = 5, 
                                      feature_names = colnames(dat_orig),
                                      verbose = FALSE)$median_auc
    
    for (j in seq(n_noisy)) {
      dt[i, j + 5] <- RfDetectionTest(dat_o = dat_orig, 
                                      dat_m = syn_noisy_miav[[j]], 
                                      n_runs = 5, 
                                      feature_names = colnames(dat_orig),
                                      verbose = FALSE)$median_auc
    }
    
    #######################################
    ## compute privacy metrics
    #######################################
    
    cat("####################### compute privacy metrics", "\n")
    
    median_dcrs[i, "hold"] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                                      dat_s = dat_hold, 
                                                      cat_variables = NULL, 
                                                      distance_type = "euclidean"))
    
    median_dcrs[i, "jf"] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                                    dat_s = syn_jf, 
                                                    cat_variables = NULL, 
                                                    distance_type = "euclidean"))
    
    median_dcrs[i, "fc"] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                                    dat_s = syn_fc, 
                                                    cat_variables = NULL, 
                                                    distance_type = "euclidean"))
    
    median_dcrs[i, "miav"] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                                      dat_s = syn_miav, 
                                                      cat_variables = NULL, 
                                                      distance_type = "euclidean"))
    
    median_dcrs[i, "smote"] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                                       dat_s = syn_smote, 
                                                       cat_variables = NULL, 
                                                       distance_type = "euclidean"))
    for (j in seq(n_noisy)) {
      median_dcrs[i, j + 5] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                                       dat_s = syn_noisy_miav[[j]], 
                                                       cat_variables = NULL, 
                                                       distance_type = "euclidean"))
    }
    
    p <- ncol(dat_orig)
    dbrls[i, "hold"] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                   dat_m = dat_hold, 
                                                   num_variables = seq(p),
                                                   sort_data = TRUE)
    
    dbrls[i, "jf"] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                 dat_m = syn_jf, 
                                                 num_variables = seq(p),
                                                 sort_data = TRUE)
    
    dbrls[i, "fc"] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                 dat_m = syn_fc, 
                                                 num_variables = seq(p),
                                                 sort_data = TRUE)
    
    dbrls[i, "miav"] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                   dat_m = syn_miav, 
                                                   num_variables = seq(p),
                                                   sort_data = TRUE)
    
    dbrls[i, "smote"] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                    dat_m = syn_smote, 
                                                    num_variables = seq(p),
                                                    sort_data = TRUE)
    
    for (j in seq(n_noisy)) {
      dbrls[i, j + 5] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                    dat_m = syn_noisy_miav[[j]], 
                                                    num_variables = seq(p),
                                                    sort_data = TRUE)
    }
    
    p <- ncol(dat_orig)
    sdids[i, "hold"] <- AverageSDID(dat_o = dat_orig, 
                                    dat_m = dat_hold, 
                                    num_variables = seq(p),
                                    sort_data = TRUE)
    
    sdids[i, "jf"] <- AverageSDID(dat_o = dat_orig, 
                                  dat_m = syn_jf, 
                                  num_variables = seq(p),
                                  sort_data = TRUE)
    
    sdids[i, "fc"] <- AverageSDID(dat_o = dat_orig, 
                                  dat_m = syn_fc, 
                                  num_variables = seq(p),
                                  sort_data = TRUE)
    
    sdids[i, "miav"] <- AverageSDID(dat_o = dat_orig, 
                                    dat_m = syn_miav, 
                                    num_variables = seq(p),
                                    sort_data = TRUE)
    
    sdids[i, "smote"] <- AverageSDID(dat_o = dat_orig, 
                                     dat_m = syn_smote, 
                                     num_variables = seq(p),
                                     sort_data = TRUE)
    
    for (j in seq(n_noisy)) {
      sdids[i, j + 5] <- AverageSDID(dat_o = dat_orig, 
                                     dat_m = syn_noisy_miav[[j]], 
                                     num_variables = seq(p),
                                     sort_data = TRUE)
    }
  }
  
  return(list(ks_test_stat = ks_test_stat,
              l2corr_dist = l2corrdist,
              energy_dist = ed,
              detection_test = dt,
              median_dcrs = median_dcrs,
              dbrls = dbrls,
              sdids = sdids))
}



GrabDataset <- function(df, task_id, split_idx, role) {
  
  idx <- df[["__task_id__"]] == task_id & 
    df[["__split__"]] == split_idx & 
    df[["__role__"]] == role
  
  df_sub <- df[idx, -c(1:4)]
  aux <- apply(df_sub, 2, function(x) sum(is.na(x)))
  cols_to_drop <- which(aux == nrow(df_sub))
  if (length(cols_to_drop) > 0) {
    df_sub <- df_sub[, -cols_to_drop]
  }
  df_sub <- as.data.frame(df_sub)
  
  return(df_sub)
}



EvaluateSyntheticData <- function(df_split,
                                  df_synth,
                                  n_runs = 5) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  ks_stat <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ks_stat) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  ed <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ed) <- dataset_names
  
  detection_tests <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(detection_tests) <- dataset_names
  
  dcrs <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dcrs) <- dataset_names
  
  dbrls <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dbrls) <- dataset_names
  
  sdids <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(sdids) <- dataset_names
  
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
      
      dat_synt <- GrabDataset(df = df_synth, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "syn")
      
      # If data synthesis failed for a dataset (so that it has 0 rows),
      # we skip the evaluation
      if (nrow(dat_synt) > 0) {
        ## add "X" to variable names
        dat_orig <- data.frame(dat_orig)
        dat_hold <- data.frame(dat_hold)
        dat_synt <- data.frame(dat_synt)
        
        aux_type <- GetVariableTypes(dat_orig)
        num_variables <- aux_type$num_variables
        cat_variables <- aux_type$cat_variables
        
        feat_names <- colnames(dat_orig)
        
        cat("ks-stat ", c(i, j), "\n")
        ks_stat[j, i] <- AverageKSTestStat(dat_o = dat_orig[, num_variables], 
                                           dat_s = dat_synt[, num_variables])
        
        cat("l2dist ", c(i, j), "\n")
        am_orig <- ComputeAssociationMatrix(dat_orig,
                                            num_variables,
                                            cat_variables)
        am_synt <- try(ComputeAssociationMatrix(dat_synt,
                                                num_variables,
                                                cat_variables), silent = TRUE)
        if (!inherits(am_synt, "try-error")) {
          l2dist[j, i] <- L2DistAssociationMatrix(am_orig, am_synt)
        }
        
        cat("ed ", c(i, j), "\n")     
        ed[j, i] <- NumericEdist(dat1_n = dat_orig[, num_variables], 
                                 dat2_n = dat_synt[, num_variables])$ed
        
        cat("detection test ", c(i, j), "\n")
        rf_syn <- RfDetectionTest(dat_o = dat_orig,
                                  dat_m = dat_synt,
                                  n_runs = n_runs,
                                  feature_names = feat_names,
                                  verbose = FALSE)
        detection_tests[j, i] <- rf_syn$median_auc
        
        cat("compute DCRs ", c(i, j), "\n")
        dcrs[j, i] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                              dat_s = dat_synt, 
                                              cat_variables = cat_variables, 
                                              distance_type = "euclidean"))
        
        cat("compute DBRL metrics ", c(i, j), "\n")
        dbrls[j, i] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                  dat_m = dat_synt, 
                                                  num_variables = num_variables,
                                                  sort_data = TRUE)
        
        cat("compute SDID metrics ", c(i, j), "\n")
        sdids[j, i] <- AverageSDID(dat_o = dat_orig, 
                                   dat_m = dat_synt, 
                                   num_variables = num_variables,
                                   k_grid = seq(0.01, 0.10, by = 0.01),
                                   sort_data = TRUE)
      }
    }
  }
  
  return(list(ks_stat = ks_stat,
              l2dist = l2dist,
              ed = ed,
              detection_tests = detection_tests,
              dcrs = dcrs,
              dbrls = dbrls,
              sdids = sdids))
}


EvaluateHoldoutData <- function(df_split,
                                n_runs = 5) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  ks_stat <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ks_stat) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  ed <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ed) <- dataset_names
  
  detection_tests <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(detection_tests) <- dataset_names
  
  dcrs <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dcrs) <- dataset_names
  
  dbrls <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dbrls) <- dataset_names
  
  sdids <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(sdids) <- dataset_names
  
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
      
      dat_synt <- dat_hold
      
      ## add "X" to variable names
      dat_orig <- data.frame(dat_orig)
      dat_hold <- data.frame(dat_hold)
      dat_synt <- data.frame(dat_synt)
      
      aux_type <- GetVariableTypes(dat_orig)
      num_variables <- aux_type$num_variables
      cat_variables <- aux_type$cat_variables
      
      feat_names <- colnames(dat_orig)
      
      cat("ks-stat ", c(i, j), "\n")
      ks_stat[j, i] <- AverageKSTestStat(dat_o = dat_orig[, num_variables], 
                                         dat_s = dat_synt[, num_variables])
      
      cat("l2dist ", c(i, j), "\n")
      am_orig <- ComputeAssociationMatrix(dat_orig,
                                          num_variables,
                                          cat_variables)
      am_synt <- ComputeAssociationMatrix(dat_synt,
                                          num_variables,
                                          cat_variables)
      l2dist[j, i] <- L2DistAssociationMatrix(am_orig, am_synt)
      
      cat("ed ", c(i, j), "\n")     
      ed[j, i] <- NumericEdist(dat1_n = dat_orig[, num_variables], 
                               dat2_n = dat_synt[, num_variables])$ed
      
      cat("detection test ", c(i, j), "\n")
      ## make sure datasets have the same number of rows
      nr <- min(c(nrow(dat_orig), nrow(dat_synt))) 
      rf_syn <- RfDetectionTest(dat_o = dat_orig[seq(nr),],
                                dat_m = dat_synt[seq(nr),],
                                n_runs = n_runs,
                                feature_names = feat_names,
                                verbose = FALSE)
      detection_tests[j, i] <- rf_syn$median_auc
      
      cat("compute DCRs ", c(i, j), "\n")
      dcrs[j, i] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                            dat_s = dat_synt, 
                                            cat_variables = cat_variables, 
                                            distance_type = "euclidean"))
      
      cat("compute DBRL metrics ", c(i, j), "\n")
      dbrls[j, i] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                dat_m = dat_synt, 
                                                num_variables = num_variables,
                                                sort_data = TRUE)
      
      cat("compute SDID metrics ", c(i, j), "\n")
      ## make sure datasets have the same number of rows
      nr <- min(c(nrow(dat_orig), nrow(dat_synt))) 
      sdids[j, i] <- AverageSDID(dat_o = dat_orig[seq(nr),], 
                                 dat_m = dat_synt[seq(nr),], 
                                 num_variables = num_variables,
                                 k_grid = seq(0.01, 0.10, by = 0.01),
                                 sort_data = TRUE)
    }
    
  }
  
  return(list(ks_stat = ks_stat,
              l2dist = l2dist,
              ed = ed,
              detection_tests = detection_tests,
              dcrs = dcrs,
              dbrls = dbrls,
              sdids = sdids))
}


EvaluateSmoteData <- function(df_split,
                              n_runs = 30,
                              k = 5) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  ks_stat <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ks_stat) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  ed <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ed) <- dataset_names
  
  detection_tests <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(detection_tests) <- dataset_names
  
  dcrs <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dcrs) <- dataset_names
  
  dbrls <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dbrls) <- dataset_names
  
  sdids <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(sdids) <- dataset_names
  
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
      
      
      aux_type <- GetVariableTypes(dat_orig)
      num_variables <- aux_type$num_variables
      cat_variables <- aux_type$cat_variables
      
      feat_names <- colnames(dat_orig)
      
      cat("ks-stat ", c(i, j), "\n")
      ks_stat[j, i] <- AverageKSTestStat(dat_o = dat_orig[, num_variables], 
                                         dat_s = dat_synt[, num_variables])
      
      cat("l2dist ", c(i, j), "\n")
      am_orig <- ComputeAssociationMatrix(dat_orig,
                                          num_variables,
                                          cat_variables)
      am_synt <- try(ComputeAssociationMatrix(dat_synt,
                                              num_variables,
                                              cat_variables), silent = TRUE)
      if (!inherits(am_synt, "try-error")) {
        l2dist[j, i] <- L2DistAssociationMatrix(am_orig, am_synt)
      }
      
      cat("ed ", c(i, j), "\n")     
      ed[j, i] <- NumericEdist(dat1_n = dat_orig[, num_variables], 
                               dat2_n = dat_synt[, num_variables])$ed
      
      cat("detection test ", c(i, j), "\n")
      rf_syn <- RfDetectionTest(dat_o = dat_orig,
                                dat_m = dat_synt,
                                n_runs = n_runs,
                                feature_names = feat_names,
                                verbose = FALSE)
      detection_tests[j, i] <- rf_syn$median_auc
      
      cat("compute DCRs ", c(i, j), "\n")
      dcrs[j, i] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                            dat_s = dat_synt, 
                                            cat_variables = cat_variables, 
                                            distance_type = "euclidean"))
      
      cat("compute DBRL metrics ", c(i, j), "\n")
      dbrls[j, i] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                                dat_m = dat_synt, 
                                                num_variables = num_variables,
                                                sort_data = TRUE)
      
      cat("compute SDID metrics ", c(i, j), "\n")
      sdids[j, i] <- AverageSDID(dat_o = dat_orig, 
                                 dat_m = dat_synt, 
                                 num_variables = num_variables,
                                 k_grid = seq(0.01, 0.10, by = 0.01),
                                 sort_data = TRUE)
    }
    
  }
  
  return(list(ks_stat = ks_stat,
              l2dist = l2dist,
              ed = ed,
              detection_tests = detection_tests,
              dcrs = dcrs,
              dbrls = dbrls,
              sdids = sdids))
}



EvaluateSyntheticDataBaseline <- function(df_split,
                                   df_synth,
                                   n_runs = 30) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  ks_stat <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ks_stat) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  ed <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ed) <- dataset_names
  
  detection_tests <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(detection_tests) <- dataset_names
  
  dcrs <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dcrs) <- dataset_names
  
  dbrls <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dbrls) <- dataset_names
  
  sdids <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(sdids) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
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
    
    dat_synt <- GrabDataset(df = df_synth, 
                            task_id = 0, 
                            split_idx = j, 
                            role = "syn")
    
    dat_orig <- data.frame(dat_orig)
    dat_hold <- data.frame(dat_hold)
    dat_synt <- data.frame(dat_synt)
    
    aux_type <- GetVariableTypes(dat_orig)
    num_variables <- aux_type$num_variables
    cat_variables <- aux_type$cat_variables
    
    feat_names <- colnames(dat_orig)
    
    #cat("ks-stat ", c(j), "\n")
    ks_stat[j, 1] <- AverageKSTestStat(dat_o = dat_orig[, num_variables], 
                                       dat_s = dat_synt[, num_variables])
    
    #cat("l2dist ", c(j), "\n")
    am_ori <- ComputeAssociationMatrix(dat_orig,
                                       num_variables,
                                       cat_variables)
    am_syn <- try(ComputeAssociationMatrix(dat_synt,
                                           num_variables,
                                           cat_variables), silent = TRUE)
    if (!inherits(am_syn, "try-error")) {
      l2dist[j, 1] <- L2DistAssociationMatrix(am_ori, am_syn)
    }
    
    #cat("ed ", c(j), "\n")     
    ed[j, 1] <- NumericEdist(dat1_n = dat_orig[, num_variables], 
                             dat2_n = dat_synt[, num_variables])$ed
    
    #cat("detection test ", c(j), "\n")
    rf_syn <- RfDetectionTest(dat_o = dat_orig,
                              dat_m = dat_synt,
                              n_runs = n_runs,
                              feature_names = feat_names,
                              verbose = FALSE)
    detection_tests[j, 1] <- rf_syn$median_auc
    
    #cat("compute DCRs ", c(j), "\n")
    dcrs[j, 1] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                          dat_s = dat_synt, 
                                          cat_variables = cat_variables, 
                                          distance_type = "euclidean"))
    
    #cat("compute DBRL metrics ", c(j), "\n")
    dbrls[j, 1] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                              dat_m = dat_synt, 
                                              num_variables = num_variables,
                                              sort_data = TRUE)
    
    #cat("compute SDID metrics ", c(j), "\n")
    sdids[j, 1] <- AverageSDID(dat_o = dat_orig, 
                               dat_m = dat_synt, 
                               num_variables = num_variables,
                               k_grid = seq(0.01, 0.10, by = 0.01),
                               sort_data = TRUE)
  }
  
  return(list(ks_stat = ks_stat,
              l2dist = l2dist,
              ed = ed,
              detection_tests = detection_tests,
              dcrs = dcrs,
              dbrls = dbrls,
              sdids = sdids))
}



## already uses scaled ED (changed ED function in "utility_functions_TabPFN_generator.R")
EvaluateHoldoutDataBaseline <- function(df_split,
                                 n_runs = 30) {
  
  dataset_names <- ""
  n_datasets <- 1
  n_splits <- length(unique(df_split$'__split__'))
  
  ks_stat <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ks_stat) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  ed <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(ed) <- dataset_names
  
  detection_tests <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(detection_tests) <- dataset_names
  
  dcrs <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dcrs) <- dataset_names
  
  dbrls <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(dbrls) <- dataset_names
  
  sdids <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(sdids) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
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
    
    dat_synt <- dat_hold
    
    dat_orig <- data.frame(dat_orig)
    dat_synt <- data.frame(dat_synt)
    
    aux_type <- GetVariableTypes(dat_orig)
    num_variables <- aux_type$num_variables
    cat_variables <- aux_type$cat_variables
    
    feat_names <- colnames(dat_orig)
    
    #cat("ks-stat ", c(j), "\n")
    ks_stat[j, 1] <- AverageKSTestStat(dat_o = dat_orig[, num_variables], 
                                       dat_s = dat_synt[, num_variables])
    
    #cat("l2dist ", c(j), "\n")
    am_ori <- ComputeAssociationMatrix(dat_orig,
                                       num_variables,
                                       cat_variables)
    am_syn <- try(ComputeAssociationMatrix(dat_synt,
                                           num_variables,
                                           cat_variables), silent = TRUE)
    if (!inherits(am_syn, "try-error")) {
      l2dist[j, 1] <- L2DistAssociationMatrix(am_ori, am_syn)
    }
    
    #cat("ed ", c(j), "\n")     
    ed[j, 1] <- NumericEdist(dat1_n = dat_orig[, num_variables], 
                             dat2_n = dat_synt[, num_variables])$ed
    
    #cat("detection test ", c(j), "\n")
    ## make sure datasets have the same number of rows
    nr <- min(c(nrow(dat_orig), nrow(dat_synt))) 
    rf_syn <- RfDetectionTest(dat_o = dat_orig[seq(nr),],
                              dat_m = dat_synt[seq(nr),],
                              n_runs = n_runs,
                              feature_names = feat_names,
                              verbose = FALSE)
    detection_tests[j, 1] <- rf_syn$median_auc
    
    #cat("compute DCRs ", c(j), "\n")
    dcrs[j, 1] <- median(ComputeScaledDCR(dat_o = dat_orig, 
                                          dat_s = dat_synt, 
                                          cat_variables = cat_variables, 
                                          distance_type = "euclidean"))
    
    #cat("compute DBRL metrics ", c(j), "\n")
    dbrls[j, 1] <- DistanceBasedRecordLinkage(dat_o = dat_orig, 
                                              dat_m = dat_synt, 
                                              num_variables = num_variables,
                                              sort_data = TRUE)
    
    #cat("compute SDID metrics ", c(j), "\n")
    ## make sure datasets have the same number of rows
    nr <- min(c(nrow(dat_orig), nrow(dat_synt))) 
    sdids[j, 1] <- AverageSDID(dat_o = dat_orig[seq(nr),], 
                               dat_m = dat_synt[seq(nr),], 
                               num_variables = num_variables,
                               k_grid = seq(0.01, 0.10, by = 0.01),
                               sort_data = TRUE)
  }
  
  return(list(ks_stat = ks_stat,
              l2dist = l2dist,
              ed = ed,
              detection_tests = detection_tests,
              dcrs = dcrs,
              dbrls = dbrls,
              sdids = sdids))
}



RunEvaluationsBaseline <- function(ds_name,
                           data_path,
                           generator_names,
                           noisy_data_path,
                           noisy_generator_names,
                           n_runs = 5) {
  df_split <- read_feather(paste0(data_path, paste0(ds_name, "_orig_hold_splits.feather")))
  
  # Create list containing generator names
  n_generators <- length(generator_names)
  n_noisy_generators <- length(noisy_generator_names)
  df_generators <- vector(mode = "list", n_generators+n_noisy_generators)
  names(df_generators) <- c(generator_names, noisy_generator_names)
  for (i in seq(n_generators)) {
    fname <- paste0(data_path, paste0(ds_name, "_syn_", generator_names[i], ".feather"))
    df_generators[[i]] <- read_feather(fname)
  }
  for (i in seq(n_noisy_generators)) {
    fname <- paste0(noisy_data_path, paste0(ds_name, "_syn_", noisy_generator_names[i], ".feather"))
    df_generators[[i+n_generators]] <- read_feather(fname)
  }  
  
  all_generator_names <- c(generator_names, noisy_generator_names)
  
  # Run evaluations
  aux_list <- vector(mode = "list", length = n_generators + n_noisy_generators + 1)
  names(aux_list) <- c("holdout", all_generator_names)
  
  
  cat("running evaluations on the holdout set", "\n")
  aux_list[[1]] <- EvaluateHoldoutDataBaseline(df_split = df_split, n_runs = n_runs)
  
  for (i in seq(n_generators+n_noisy_generators)) {
    cat("running evaluations for: ", all_generator_names[i], "\n")
    aux_list[[i + 1]] <- EvaluateSyntheticDataBaseline(df_split = df_split,
                                                df_synth = df_generators[[i]],
                                                n_runs = n_runs)
  }
  
  
  # Organize outputs by evaluation metric
  p <- length(aux_list)
  nms <- names(aux_list)
  
  ks_stat <- aux_list[[1]]$ks_stat
  for (i in seq(2, p)) {
    ks_stat <- cbind(ks_stat, aux_list[[i]]$ks_stat)
  }
  colnames(ks_stat) <- nms
  
  l2dist <- aux_list[[1]]$l2dist
  for (i in seq(2, p)) {
    l2dist <- cbind(l2dist, aux_list[[i]]$l2dist)
  }
  colnames(l2dist) <- nms
  
  ed <- aux_list[[1]]$ed
  for (i in seq(2, p)) {
    ed <- cbind(ed, aux_list[[i]]$ed)
  }
  colnames(ed) <- nms
  
  dt <- aux_list[[1]]$detection_tests
  for (i in seq(2, p)) {
    dt <- cbind(dt, aux_list[[i]]$detection_tests)
  }
  colnames(dt) <- nms
  
  dcrs <- aux_list[[1]]$dcrs
  for (i in seq(2, p)) {
    dcrs <- cbind(dcrs, aux_list[[i]]$dcrs)
  }
  colnames(dcrs) <- nms
  
  dbrls <- aux_list[[1]]$dbrls
  for (i in seq(2, p)) {
    dbrls <- cbind(dbrls, aux_list[[i]]$dbrls)
  }
  colnames(dbrls) <- nms
  
  sdids <- aux_list[[1]]$sdids
  for (i in seq(2, p)) {
    sdids <- cbind(sdids, aux_list[[i]]$sdids)
  }
  colnames(sdids) <- nms
  
  return(list(ks_stat = ks_stat,
              l2dist = l2dist,
              ed = ed,
              detection_tests = dt,
              dcrs = dcrs,
              dbrls = dbrls,
              sdids = sdids))
}







EvaluateSyntheticDataCat <- function(df_split,
                                     df_synth) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  kl_dive <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(kl_dive) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
  ## n_datasets
  for (i in seq(n_datasets)) {
    cat("dataset ", i, "\n")
    
    for (j in seq(n_splits)) {
      cat("split ", j, "\n")
      
      dat_orig <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "orig")
      
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
        dat_orig <- data.frame(dat_orig)
        dat_hold <- data.frame(dat_hold)
        dat_synt <- data.frame(dat_synt)
        
        aux_type <- GetVariableTypes(dat_orig)
        num_variables <- aux_type$num_variables
        cat_variables <- aux_type$cat_variables
        
        feat_names <- colnames(dat_orig)
        
        cat("kl-divergence ", c(i, j), "\n")
        kl_dive[j, i] <- AverageKLDivergenceCat(dat_o = dat_orig, 
                                                dat_s = dat_synt)
        
        cat("l2dist ", c(i, j), "\n")
        am_orig <- ComputeAssociationMatrix(dat_orig,
                                            num_variables,
                                            cat_variables)
        am_synt <- ComputeAssociationMatrix(dat_synt,
                                            num_variables,
                                            cat_variables)
        l2dist[j, i] <- L2DistAssociationMatrix(am_orig, am_synt)
        
      }
    }
  }
  
  return(list(kl_dive = kl_dive,
              l2dist = l2dist))
}



EvaluateHoldoutDataCat <- function(df_split) {
  
  dataset_names <- unique(df_split$'__dataset__')
  n_datasets <- length(dataset_names)
  n_splits <- length(unique(df_split$'__split__'))
  
  kl_dive <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(kl_dive) <- dataset_names
  
  l2dist <- data.frame(matrix(NA, n_splits, n_datasets))
  colnames(l2dist) <- dataset_names
  
  task_ids <- unique(df_split$'__task_id__')
  n_datasets <- length(task_ids)
  
  ## n_datasets
  for (i in seq(n_datasets)) {
    cat("dataset ", i, "\n")
    
    for (j in seq(n_splits)) {
      cat("split ", j, "\n")
      
      dat_orig <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "orig")
      
      dat_hold <- GrabDataset(df = df_split, 
                              task_id = task_ids[i], 
                              split_idx = j, 
                              role = "hold")
      
      dat_synt <- dat_hold
      
      # If data synthesis failed for a dataset (so that it has 0 rows),
      # we skip the evaluation
      if (nrow(dat_synt) > 0) {
        ## add "X" to variable names
        dat_orig <- data.frame(dat_orig)
        dat_hold <- data.frame(dat_hold)
        dat_synt <- data.frame(dat_synt)
        
        aux_type <- GetVariableTypes(dat_orig)
        num_variables <- aux_type$num_variables
        cat_variables <- aux_type$cat_variables
        
        feat_names <- colnames(dat_orig)
        
        cat("kl-divergence ", c(i, j), "\n")
        kl_dive[j, i] <- AverageKLDivergenceCat(dat_o = dat_orig, 
                                                dat_s = dat_synt)
        
        cat("l2dist ", c(i, j), "\n")
        am_orig <- ComputeAssociationMatrix(dat_orig,
                                            num_variables,
                                            cat_variables)
        am_synt <- ComputeAssociationMatrix(dat_synt,
                                            num_variables,
                                            cat_variables)
        l2dist[j, i] <- L2DistAssociationMatrix(am_orig, am_synt)
        
      }
    }
  }
  
  return(list(kl_dive = kl_dive,
              l2dist = l2dist))
}






##########################################
## Plotting functions
##########################################

MarginalDensityPlotsQC <- function(var_idx,
                                   dat_real,
                                   dat_synt,
                                   leg_pos = "topright") {
  
  densi_real <- density(dat_real[, var_idx])
  densi_synt <- density(dat_synt[, var_idx])
  
  my_ylim <- c(0, max(densi_real$y, densi_synt$y))
  lab_at <- min(densi_real$x)
  
  plot(densi_real$x, densi_real$y, type = "l", 
       xlab = "", ylab = "density", 
       main = "", lwd = 2,
       col = "blue", ylim = my_ylim)
  lines(densi_synt$x, densi_synt$y, type = "l", col = "red")
  legend(leg_pos, legend = c("synth", "real"), 
         text.col = c("red", "blue"), bty = "n")
  
}


MarginalDensityPlotsQC2 <- function(var_idx,
                                    dat_real,
                                    dat_synt,
                                    leg_pos = "topright",
                                    method_name,
                                    method_color,
                                    main = "",
                                    adjust = 1) {
  
  densi_real <- density(dat_real[, var_idx], adjust = adjust)
  densi_synt <- density(dat_synt[, var_idx], adjust = adjust)
  
  my_ylim <- c(0, max(densi_real$y, densi_synt$y))
  lab_at <- min(densi_real$x)
  
  plot(densi_real$x, densi_real$y, type = "l", 
       xlab = "variable values", ylab = "density", 
       main = main, lwd = 2,
       col = "black", ylim = my_ylim)
  lines(densi_synt$x, densi_synt$y, type = "l", col = method_color)
  legend(leg_pos, legend = c("original", method_name), 
         text.col = c("black", method_color), bty = "n")
  
}


MarginalDensityPlotsQC3 <- function(var_idx,
                                    X,
                                    M,
                                    leg_pos = "topright",
                                    main = "") {
  
  densi_X <- density(X[, var_idx])
  densi_M <- density(M[, var_idx])
  
  my_ylim <- c(0, max(densi_X$y, densi_M$y))
  
  my_xlim <- c(min(densi_X$x, densi_M$x), 
               max(densi_X$x, densi_M$x))
  
  plot(densi_X$x, densi_X$y, type = "l", 
       xlab = "variable values", ylab = "density", 
       main = main, lwd = 1,
       col = "black", ylim = my_ylim, xlim = my_xlim)
  lines(densi_M$x, densi_M$y, type = "l", col = "red")
  legend(leg_pos, legend = c(bquote(italic(X[.(var_idx)])),
                             bquote(italic(M[.(var_idx)]))), 
         text.col = c("black", "red"), bty = "n")
  
}



MarginalDensityPlotsQCList <- function(var_idx,
                                       dat_real,
                                       dat_synt_list,
                                       leg_pos = "topright",
                                       methods_names,
                                       methods_color,
                                       main = "") {
  
  densi_real <- density(dat_real[, var_idx])
  
  n_methods <- length(dat_synt_list)
  densi_synt <- vector(mode = "list", length = n_methods)
  y_max <- rep(NA, n_methods)
  for (i in seq(n_methods)) {
    densi_synt[[i]] <- density(dat_synt_list[[i]][, var_idx])
    y_max[i] <- max(densi_synt[[i]]$y)
  }
  
  my_ylim <- c(0, max(densi_real$y, y_max))
  lab_at <- min(densi_real$x)
  plot(densi_real$x, densi_real$y, type = "l", 
       xlab = "variable values", ylab = "density", 
       main = main, lwd = 2,
       col = "black", ylim = my_ylim)
  for (i in seq(n_methods)) {
    lines(densi_synt[[i]]$x, densi_synt[[i]]$y, type = "l", col = methods_color[i])
  }
  legend(leg_pos, legend = c("original", methods_names), 
         text.col = c("black", methods_color), bty = "n")
  
}


############################


PoolResults <- function(output_list,
                        keep,
                        metric_name) {
  n <- length(output_list)
  out <- output_list[[1]][[metric_name]][, keep]
  for (i in seq(2, n)) {
    tmp <- output_list[[i]][[metric_name]][, keep]
    out <- rbind(out, tmp)
  }
  
  return(out)
}


PoolResultsReal <- function(output_list,
                            metric_name) {
  n <- length(output_list)
  tmp <- unlist(output_list[[1]][[metric_name]])
  out <- matrix(NA, length(tmp), n)
  colnames(out) <- names(output_list)
  out[, 1] <- tmp
  
  for (i in seq(2, n)) {
    out[, i] <- unlist(output_list[[i]][[metric_name]])
  }
  
  return(out)
}


PoolResultsB <- function(output_list,
                         metric_name) {
  n <- length(output_list)
  out <- output_list[[1]][[metric_name]]
  for (i in seq(2, n)) {
    tmp <- output_list[[i]][[metric_name]]
    out <- rbind(out, tmp)
  }
  
  return(out)
}


ReorganizeRealResults <- function(output_list, sel_data, metric_names) {
  
  method_names <- names(output_list)
  num_methods <- length(method_names)
  
  dataset_names <- colnames(output_list[[1]][[1]])
  num_datasets <- length(dataset_names)
  
  num_metrics <- length(metric_names)
  
  num_repli <- nrow(output_list[[1]][[1]])
  
  out <- vector(mode = "list", length = num_metrics)
  names(out) <- metric_names
  
  for (i in seq(num_metrics)) {
    out_metrics <- matrix(NA, num_repli, num_methods)
    colnames(out_metrics) <- method_names
    for (j in seq(num_methods)) {
      out_metrics[, j] <- output_list[[j]][[i]][, sel_data]
    }
    out[[i]] <- out_metrics
  }
  
  return(out)
}


PlotRow <- function(out, my_rho, i, my_line, my_cex) {
  boxplot(out$ks_test_stat[, keep], 
          main = bquote("KS, " ~abs(italic(rho)) == .(my_rho)),
          las = my_las,
          ylab = "ave. KS-statistic", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(a", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$l2corr_dist[, keep], 
          main = bquote("L2D, " ~abs(italic(rho)) == .(my_rho)), 
          las = my_las, 
          ylab = "L2 dist. between assoc. matrices", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(b", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$detection_test[, keep], 
          main = bquote("DT, " ~abs(italic(rho)) == .(my_rho)), 
          las = my_las, 
          ylab = "detection test AUROC", outline = FALSE, names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(c", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  
  boxplot(out$median_dcrs[, keep], 
          main = bquote("DCR, " ~abs(italic(rho)) == .(my_rho)), 
          las = my_las, 
          ylab = "median of DCR distribution", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(d", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$dbrls[, keep], 
          main = bquote("SDBRL, " ~abs(italic(rho)) == .(my_rho)), 
          las = my_las, 
          ylab = "sorted DBRL", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(e", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$sdids[, keep], 
          main = bquote("SSDID, " ~abs(italic(rho)) == .(my_rho)), 
          las = my_las, 
          ylab = "sorted SDID", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(f", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
}




PlotRowReal <- function(out, i, dataset_name, my_line, my_cex) {
  keep <- seq(ncol(out[[1]]))
  boxplot(out$ks_test_stat[, keep], 
          main = paste0("KS, ", dataset_name),
          las = my_las,
          ylab = "ave. KS-statistic", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(a", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$l2corr_dist[, keep], 
          main = paste0("L2D, ", dataset_name), 
          las = my_las, 
          ylab = "L2 dist. between assoc. matrices", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(b", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$detection_test[, keep], 
          main = paste0("DT, ", dataset_name), 
          las = my_las, 
          ylab = "AUROC", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(c", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$median_dcrs[, keep], 
          main = paste0("DCR, ", dataset_name), 
          las = my_las, 
          ylab = "median of DCR distribution", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(d", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$dbrls[, keep], 
          main = paste0("SDBRL, ", dataset_name), 
          las = my_las, 
          ylab = "sorted DBRL", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(e", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$sdids[, keep], 
          main = paste0("SSDID, ", dataset_name), 
          las = my_las, 
          ylab = "sorted SDID", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(f", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
}



PlotRowRealB <- function(out, i, dataset_name, my_line, my_cex, 
                         keep, method_names, methods_color) {
  nms2 <- method_names
  boxplot(out$ks_stat[, keep], 
          main = paste0("KS, ", dataset_name),
          las = my_las,
          ylab = "ave. KS-statistic", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(a", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$l2dist[, keep], 
          main = paste0("L2D, ", dataset_name), 
          las = my_las, 
          ylab = "L2 dist. between assoc. matrices", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(b", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$detection_test[, keep], 
          main = paste0("DT, ", dataset_name), 
          las = my_las, 
          ylab = "AUROC", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(c", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$dcrs[, keep], 
          main = paste0("DCR, ", dataset_name), 
          las = my_las, 
          ylab = "median of DCR distribution", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(d", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$dbrls[, keep], 
          main = paste0("SDBRL, ", dataset_name), 
          las = my_las, 
          ylab = "sorted DBRL", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(e", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
  ####
  boxplot(out$sdids[, keep], 
          main = paste0("SSDID, ", dataset_name), 
          las = my_las, 
          ylab = "sorted SDID", names = nms2, col = "white", 
          border = methods_color)
  mtext(paste0("(f", i, ")"), side = 3, adj = 0, line = my_line, cex = my_cex)
}





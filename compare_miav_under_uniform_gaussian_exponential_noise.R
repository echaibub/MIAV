
source("utility_functions_for_miav_tabpfn_iclr.R")


GenerateMaximalInformationAuxiliaryVariable <- function(x) {
  n <- length(x)
  
  if (noise_distribution == "uniform") {
    m <- runif(n)
  }
  if (noise_distribution == "gaussian") {
    m <- rnorm(n)
  }
  if (noise_distribution == "exponential") {
    m <- rexp(n)
  }
  
  m <- sort(m)
  
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



# To run TabPFN in R: 
# First in a terminal, create a virtual environment, activate it, and install 
# TabPFN. Then download the "generate_tabpfn_predictions.py" script.
#
# In R load reticulate and run the python script, which make classifiers 
# and regression models based on TabPFN available in R.
library(reticulate)
use_virtualenv("~/TabPFN/venv")
source_python("~/TabPFN/venv/generate_tabpfn_predictions.py")

manus_path <- "" # path to the folder storing the figures


######################################################
######################################################

beta_pars_list <- list()
beta_pars_list[[1]] <- c(10, 10)
beta_pars_list[[2]] <- c(0.5, 5)
beta_pars_list[[3]] <- c(30, 3)
beta_pars_list[[4]] <- c(0.5, 0.5)
beta_pars_list[[5]] <- c(3, 10)

n <- 1000


#########################################
## uniform noise
#########################################

noise_distribution <- "uniform"

rho <- -0.75

set.seed(123)
X <- SimulateCorrelatedBetaData(n = n, 
                                rho = rho, 
                                beta_pars_list = beta_pars_list)
set.seed(1234)
X[, 2] <- X[sample(n), 2]


M <- ComputeAuxiliaryVariables(X)
colnames(M) <- paste0("M", seq(5))


leg_positions3 <- c("topright", "topright", "topleft", "bottom", "topright")


## Figure 25

par(mfrow = c(2, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
for (i in seq(5)) {
  MarginalDensityPlotsQC3(var_idx = i,
                          X = X,
                          M = M,
                          leg_pos = leg_positions3[i],
                          main = bquote(M[.(i)] ~ "," ~ X[.(i)] ~ " distr."))
  mtext(paste0("(", letters[i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
####
for (i in seq(5)) {
  plot(X[, i], M[, i],
       xlab = bquote(italic(X[.(i)])),
       ylab = bquote(italic(M[.(i)])),
       main = bquote(M[.(i)] ~ " vs " ~ X[.(i)]),
       cex = 0.5)
  mtext(paste0("(", letters[5+i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, mgp = c(3, 1, 0))


set.seed(123)
miav_unif_1 <- MiavTabPFNGenerator(X = X)

set.seed(1234)
miav_unif_2 <- MiavTabPFNGenerator(X = X)

set.seed(12345)
miav_unif_3 <- MiavTabPFNGenerator(X = X)

leg_positions <- c("bottom", "topright", "topleft", "bottom", "topright")

# Figure 28

par(mfrow = c(3, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_unif_1,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 1)"))
  mtext(paste0("(", letters[j], ")"), side = 3, adj = 0)
}
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_unif_2,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 2)"))
  mtext(paste0("(", letters[j+5], ")"), side = 3, adj = 0)
}
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_unif_3,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 3)"))
  mtext(paste0("(", letters[j+10], ")"), side = 3, adj = 0)
}




#########################################
## gaussian noise
#########################################

noise_distribution <- "gaussian"

rho <- -0.75

set.seed(123)
X <- SimulateCorrelatedBetaData(n = n, 
                                rho = rho, 
                                beta_pars_list = beta_pars_list)
set.seed(1234)
X[, 2] <- X[sample(n), 2]


M <- ComputeAuxiliaryVariables(X)
colnames(M) <- paste0("M", seq(5))

leg_positions3 <- c("topright", "topright", "topright", "topright", "topright")


# Figure 26

par(mfrow = c(2, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
for (i in seq(5)) {
  MarginalDensityPlotsQC3(var_idx = i,
                          X = X,
                          M = M,
                          leg_pos = leg_positions3[i],
                          main = bquote(M[.(i)] ~ "," ~ X[.(i)] ~ " distr."))
  mtext(paste0("(", letters[i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
####
for (i in seq(5)) {
  plot(X[, i], M[, i],
       xlab = bquote(italic(X[.(i)])),
       ylab = bquote(italic(M[.(i)])),
       main = bquote(M[.(i)] ~ " vs " ~ X[.(i)]),
       cex = 0.5)
  mtext(paste0("(", letters[5+i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, mgp = c(3, 1, 0))


set.seed(123)
miav_gauss_1 <- MiavTabPFNGenerator(X = X)

set.seed(1234)
miav_gauss_2 <- MiavTabPFNGenerator(X = X)

set.seed(12345)
miav_gauss_3 <- MiavTabPFNGenerator(X = X)


# Figure 29

par(mfrow = c(3, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_gauss_1,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 1)"))
  mtext(paste0("(", letters[j+6], ")"), side = 3, adj = 0)
}
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_gauss_2,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 2)"))
  mtext(paste0("(", letters[j+5], ")"), side = 3, adj = 0)
}
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_gauss_3,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 3)"))
  mtext(paste0("(", letters[j+10], ")"), side = 3, adj = 0)
}



##############################################
## exponential noise
##############################################

noise_distribution <- "exponential"

rho <- -0.75

set.seed(123)
X <- SimulateCorrelatedBetaData(n = n, 
                                rho = rho, 
                                beta_pars_list = beta_pars_list)
set.seed(1234)
X[, 2] <- X[sample(n), 2]


M <- ComputeAuxiliaryVariables(X)
colnames(M) <- paste0("M", seq(5))

leg_positions3 <- c("topright", "topright", "topright", "topright", "topright")

# Figure 27

par(mfrow = c(2, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
for (i in seq(5)) {
  MarginalDensityPlotsQC3(var_idx = i,
                          X = X,
                          M = M,
                          leg_pos = leg_positions3[i],
                          main = bquote(M[.(i)] ~ "," ~ X[.(i)] ~ " distr."))
  mtext(paste0("(", letters[i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
####
for (i in seq(5)) {
  plot(X[, i], M[, i],
       xlab = bquote(italic(X[.(i)])),
       ylab = bquote(italic(M[.(i)])),
       main = bquote(M[.(i)] ~ " vs " ~ X[.(i)]),
       cex = 0.5)
  mtext(paste0("(", letters[5+i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, mgp = c(3, 1, 0))


set.seed(123)
miav_expon_1 <- MiavTabPFNGenerator(X = X)

set.seed(1234)
miav_expon_2 <- MiavTabPFNGenerator(X = X)

set.seed(12345)
miav_expon_3 <- MiavTabPFNGenerator(X = X)

# Figure 30

par(mfrow = c(3, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_expon_1,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 1)"))
  mtext(paste0("(", letters[j], ")"), side = 3, adj = 0)
}
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_expon_2,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 2)"))
  mtext(paste0("(", letters[j+5], ")"), side = 3, adj = 0)
}
for (j in seq(5)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = X,
                          dat_synt = miav_expon_3,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = "red",
                          main = bquote("    " ~ italic(X[.(j)]) ~ "(repl. 3)"))
  mtext(paste0("(", letters[j+10], ")"), side = 3, adj = 0)
}



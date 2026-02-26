
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


manus_path <- ""


########################################################
## Set shape parameters for the beta distributions 
########################################################

beta_pars_list <- list()
beta_pars_list[[1]] <- c(10, 10)
beta_pars_list[[2]] <- c(0.5, 5)
beta_pars_list[[3]] <- c(30, 3)
beta_pars_list[[4]] <- c(0.5, 0.5)
beta_pars_list[[5]] <- c(3, 10)

n <- 1000

##########################################################################
# Generate original, holdout and synthetic datasets for rho = -0.95
##########################################################################

set.seed(123456789)
dat_orig_0.95 <- SimulateCorrelatedBetaData(n = n, 
                                           rho = -0.95, 
                                           beta_pars_list = beta_pars_list)

dat_hold_0.95 <- SimulateCorrelatedBetaData(n = n, 
                                           rho = -0.95, 
                                           beta_pars_list = beta_pars_list)

set.seed(1234)
dat_orig_0.95[, 2] <- dat_orig_0.95[sample(n), 2]
dat_hold_0.95[, 2] <- dat_hold_0.95[sample(n), 2]

set.seed(12345)
syn_jf_0.95 <- JointFactorizationTabPFNGenerator(X = dat_orig_0.95[sample(n),])
syn_fc_0.95 <- FullConditionalsTabPFNGenerator(X = dat_orig_0.95[sample(n),])
syn_miav_0.95 <- MiavTabPFNGenerator(X = dat_orig_0.95[sample(n),])
save(dat_orig_0.95, dat_hold_0.95, syn_jf_0.95, syn_fc_0.95, syn_miav_0.95, 
     file = "example1_tabpfn_models_neg0.95_rho.RData", compress = TRUE)


##########################################################################
# Generate original, holdout and synthetic datasets for rho = -0.75
##########################################################################

set.seed(123456789)
dat_orig_0.75 <- SimulateCorrelatedBetaData(n = n, 
                                            rho = -0.75, 
                                            beta_pars_list = beta_pars_list)

dat_hold_0.75 <- SimulateCorrelatedBetaData(n = n, 
                                            rho = -0.75, 
                                            beta_pars_list = beta_pars_list)

set.seed(1234)
dat_orig_0.75[, 2] <- dat_orig_0.75[sample(n), 2]
dat_hold_0.75[, 2] <- dat_hold_0.75[sample(n), 2]

set.seed(12345)
syn_jf_0.75 <- JointFactorizationTabPFNGenerator(X = dat_orig_0.75[sample(n),])
syn_fc_0.75 <- FullConditionalsTabPFNGenerator(X = dat_orig_0.75[sample(n),])
syn_miav_0.75 <- MiavTabPFNGenerator(X = dat_orig_0.75[sample(n),])
save(dat_orig_0.75, dat_hold_0.75, syn_jf_0.75, syn_fc_0.75, syn_miav_0.75, 
     file = "example1_tabpfn_models_neg0.75_rho.RData", compress = TRUE)



##########################################################################
# Generate original, holdout and synthetic datasets for rho = -0.5
##########################################################################

set.seed(123456789)
dat_orig_0.5 <- SimulateCorrelatedBetaData(n = n, 
                                           rho = -0.5, 
                                           beta_pars_list = beta_pars_list)

dat_hold_0.5 <- SimulateCorrelatedBetaData(n = n, 
                                           rho = -0.5, 
                                           beta_pars_list = beta_pars_list)

set.seed(1234)
dat_orig_0.5[, 2] <- dat_orig_0.5[sample(n), 2]
dat_hold_0.5[, 2] <- dat_hold_0.5[sample(n), 2]

set.seed(12345)
syn_jf_0.5 <- JointFactorizationTabPFNGenerator(X = dat_orig_0.5[sample(n),])
syn_fc_0.5 <- FullConditionalsTabPFNGenerator(X = dat_orig_0.5[sample(n),])
syn_miav_0.5 <- MiavTabPFNGenerator(X = dat_orig_0.5[sample(n),])
save(dat_orig_0.5, dat_hold_0.5, syn_jf_0.5, syn_fc_0.5, syn_miav_0.5, 
     file = "example1_tabpfn_models_neg0.5_rho.RData", compress = TRUE)


##########################################################################
# Generate original, holdout and synthetic datasets for rho = -0.25
##########################################################################

set.seed(123456789)
dat_orig_0.25 <- SimulateCorrelatedBetaData(n = n, 
                                            rho = -0.25, 
                                            beta_pars_list = beta_pars_list)

dat_hold_0.25 <- SimulateCorrelatedBetaData(n = n, 
                                            rho = -0.25, 
                                            beta_pars_list = beta_pars_list)

set.seed(1234)
dat_orig_0.25[, 2] <- dat_orig_0.25[sample(n), 2]
dat_hold_0.25[, 2] <- dat_hold_0.25[sample(n), 2]

set.seed(12345)
syn_jf_0.25 <- JointFactorizationTabPFNGenerator(X = dat_orig_0.25[sample(n),])
syn_fc_0.25 <- FullConditionalsTabPFNGenerator(X = dat_orig_0.25[sample(n),])
syn_miav_0.25 <- MiavTabPFNGenerator(X = dat_orig_0.25[sample(n),])
save(dat_orig_0.25, dat_hold_0.25, syn_jf_0.25, syn_fc_0.25, syn_miav_0.25, 
     file = "example1_tabpfn_models_neg0.25_rho.RData", compress = TRUE)



##########################################################################
# Generate original, holdout and synthetic datasets for rho = 0
##########################################################################

set.seed(123456789)
dat_orig_0 <- SimulateCorrelatedBetaData(n = n, 
                                            rho = 0, 
                                            beta_pars_list = beta_pars_list)

dat_hold_0 <- SimulateCorrelatedBetaData(n = n, 
                                            rho = 0, 
                                            beta_pars_list = beta_pars_list)

set.seed(1234)
dat_orig_0[, 2] <- dat_orig_0[sample(n), 2]
dat_hold_0[, 2] <- dat_hold_0[sample(n), 2]

set.seed(12345)
syn_jf_0 <- JointFactorizationTabPFNGenerator(X = dat_orig_0[sample(n),])
syn_fc_0 <- FullConditionalsTabPFNGenerator(X = dat_orig_0[sample(n),])
syn_miav_0 <- MiavTabPFNGenerator(X = dat_orig_0[sample(n),])
save(dat_orig_0, dat_hold_0, syn_jf_0, syn_fc_0, syn_miav_0, 
     file = "example1_tabpfn_models_0_rho.RData", compress = TRUE)


# load outputs
load("example1_tabpfn_models_neg0.95_rho.RData")
load("example1_tabpfn_models_neg0.75_rho.RData")
load("example1_tabpfn_models_neg0.5_rho.RData")
load("example1_tabpfn_models_neg0.25_rho.RData")
load("example1_tabpfn_models_0_rho.RData")


#############################################
## Generate Figure 1 in the main text
#############################################

dat_orig <- dat_orig_0.95
dat_hold <- dat_hold_0.95
dat_synt_list <- list(syn_jf_0.95, syn_fc_0.95, syn_miav_0.95)
syn_jf <- dat_synt_list[[1]]
syn_fc <- dat_synt_list[[2]]
syn_miav <- dat_synt_list[[3]]

methods_names <- c("JF", "FC", "MIAV")
methods_color <- c("darkorange", "blue", "red")

leg_positions <- c("topleft", "topright", "topleft", "bottom", "topright")


par(mfrow = c(1, 5), mar = c(3, 2.5, 1, 0.25) + 0.1, mgp = c(1.75, 0.75, 0))
for (i in seq(5)) {
  MarginalDensityPlotsQCList(var_idx = i,
                             dat_real = dat_orig,
                             dat_synt_list = dat_synt_list,
                             leg_pos = leg_positions[i],
                             methods_names = methods_names,
                             methods_color = methods_color,
                             main = bquote(italic(X[.(i)])))
  mtext(paste0("(", letters[i], ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, mgp = c(3, 1, 0))



#############################################
## Generate Figure 2 in the main text
#############################################

var_idx <- 4

leg_positions2 <- c("bottom", "bottom", "bottom", "topleft", "topleft")

par(mfrow = c(1, 5), mar = c(3, 2.5, 1, 0.25) + 0.1, mgp = c(1.75, 0.75, 0))
dat_synt_list_0.95 <- list(syn_jf_0.95, syn_fc_0.95, syn_miav_0.95)
MarginalDensityPlotsQCList(var_idx = var_idx,
                           dat_real = dat_orig_0.95,
                           dat_synt_list = dat_synt_list_0.95,
                           leg_pos = leg_positions2[1],
                           methods_names = methods_names,
                           methods_color = methods_color,
                           main = expression(rho == -0.95))
mtext("(a)", side = 3, adj = 0)
dat_synt_list_0.75 <- list(syn_jf_0.75, syn_fc_0.75, syn_miav_0.95)
MarginalDensityPlotsQCList(var_idx = var_idx,
                           dat_real = dat_orig_0.75,
                           dat_synt_list = dat_synt_list_0.75,
                           leg_pos = leg_positions2[2],
                           methods_names = methods_names,
                           methods_color = methods_color,
                           main = expression(rho == -0.75))
mtext("(b)", side = 3, adj = 0)
dat_synt_list_0.5 <- list(syn_jf_0.5, syn_fc_0.5, syn_miav_0.5)
MarginalDensityPlotsQCList(var_idx = var_idx,
                           dat_real = dat_orig_0.5,
                           dat_synt_list = dat_synt_list_0.5,
                           leg_pos = leg_positions2[3],
                           methods_names = methods_names,
                           methods_color = methods_color,
                           main = expression(rho == -0.5))
mtext("(c)", side = 3, adj = 0)
dat_synt_list_0.25 <- list(syn_jf_0.25, syn_fc_0.25, syn_miav_0.25)
MarginalDensityPlotsQCList(var_idx = var_idx,
                           dat_real = dat_orig_0.25,
                           dat_synt_list = dat_synt_list_0.25,
                           leg_pos = leg_positions2[4],
                           methods_names = methods_names,
                           methods_color = methods_color,
                           main = expression(rho == -0.25))
mtext("(d)", side = 3, adj = 0)
dat_synt_list_0 <- list(syn_jf_0, syn_fc_0, syn_miav_0)
MarginalDensityPlotsQCList(var_idx = var_idx,
                           dat_real = dat_orig_0,
                           dat_synt_list = dat_synt_list_0,
                           leg_pos = leg_positions2[5],
                           methods_names = methods_names,
                           methods_color = methods_color,
                           main = expression(rho == 0))
mtext("(e)", side = 3, adj = 0)
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, mgp = c(3, 1, 0))



###############################################################
## Generate Supplementary Figure with rho = -0.95 (Figure 5)
###############################################################

dat_orig <- dat_orig_0.95
dat_hold <- dat_hold_0.95
dat_synt_list <- list(syn_jf_0.95, syn_fc_0.95, syn_miav_0.95)
syn_jf <- dat_synt_list[[1]]
syn_fc <- dat_synt_list[[2]]
syn_miav <- dat_synt_list[[3]]


am_orig <- ComputeAssociationMatrix(dat_orig,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_hold <- ComputeAssociationMatrix(dat_hold,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_jf <- ComputeAssociationMatrix(syn_jf,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_fc <- ComputeAssociationMatrix(syn_fc,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_miav <- ComputeAssociationMatrix(syn_miav,
                                    num_variables = seq(5),
                                    cat_variables = NULL)


my_mar <- c(2.5, 2.5, 2, 0.5)
p <- 5

my_mar2 <- c(2, 0.2, 1.2, 0.2)

nms <- colnames(dat_orig)
par(mfrow = c(5, 6), mar = my_mar, mgp = c(1.5, 0.5, 0))
plot.new()
text(0.5, 0.5, expression(rho == -0.95), cex = 1.5)
corrplot(am_orig, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "original", line = 1)
mtext("(a)", side = 3, adj = 0)
corrplot(am_jf, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "JF", line = 1)
mtext("(b)", side = 3, adj = 0)
corrplot(am_fc, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "FC", line = 1)
mtext("(c)", side = 3, adj = 0)
corrplot(am_miav, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "MIAV", line = 1)
mtext("(d)", side = 3, adj = 0)
corrplot(am_hold, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "holdout", line = 1)
mtext("(e)", side = 3, adj = 0)
############################################
corrplot(am_orig - am_jf, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. JF", line = 1)
mtext("(f)", side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_jf,
                          leg_pos = leg_positions[j],
                          method_name = "JF",
                          method_color = methods_color[1],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+6], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_fc, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[12], ")"), side = 3, adj = 0)
title(main = "delta corr. FC", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_fc,
                          leg_pos = leg_positions[j],
                          method_name = "FC",
                          method_color = methods_color[2],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+12], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_miav, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[18], ")"), side = 3, adj = 0)
title(main = "delta corr. MIAV", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_miav,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = methods_color[3],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+18], ")"), side = 3, adj = 0)
}
###############################################
corrplot(am_orig - am_hold, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. hldt", line = 1)
mtext(paste0("(", letters[26], ")"), side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = dat_hold,
                          leg_pos = leg_positions[j],
                          method_name = "holdout",
                          method_color = "green",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(z", j, ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)



################################################################
## Generate Supplementary Figure with rho = -0.75 (Figure 6)
################################################################

dat_orig <- dat_orig_0.75
dat_hold <- dat_hold_0.75
dat_synt_list <- list(syn_jf_0.75, syn_fc_0.75, syn_miav_0.75)
syn_jf <- dat_synt_list[[1]]
syn_fc <- dat_synt_list[[2]]
syn_miav <- dat_synt_list[[3]]


am_orig <- ComputeAssociationMatrix(dat_orig,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_hold <- ComputeAssociationMatrix(dat_hold,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_jf <- ComputeAssociationMatrix(syn_jf,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_fc <- ComputeAssociationMatrix(syn_fc,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_miav <- ComputeAssociationMatrix(syn_miav,
                                    num_variables = seq(5),
                                    cat_variables = NULL)


my_mar <- c(2.5, 2.5, 2, 0.5)
p <- 5

my_mar2 <- c(2, 0.2, 1.2, 0.2)

nms <- colnames(dat_orig)
par(mfrow = c(5, 6), mar = my_mar, mgp = c(1.5, 0.5, 0))
plot.new()
text(0.5, 0.5, expression(rho == -0.75), cex = 1.5)
corrplot(am_orig, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "original", line = 1)
mtext("(a)", side = 3, adj = 0)
corrplot(am_jf, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "JF", line = 1)
mtext("(b)", side = 3, adj = 0)
corrplot(am_fc, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "FC", line = 1)
mtext("(c)", side = 3, adj = 0)
corrplot(am_miav, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "MIAV", line = 1)
mtext("(d)", side = 3, adj = 0)
corrplot(am_hold, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "holdout", line = 1)
mtext("(e)", side = 3, adj = 0)
############################################
corrplot(am_orig - am_jf, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. JF", line = 1)
mtext("(f)", side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_jf,
                          leg_pos = leg_positions[j],
                          method_name = "JF",
                          method_color = methods_color[1],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+6], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_fc, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[12], ")"), side = 3, adj = 0)
title(main = "delta corr. FC", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_fc,
                          leg_pos = leg_positions[j],
                          method_name = "FC",
                          method_color = methods_color[2],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+12], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_miav, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[18], ")"), side = 3, adj = 0)
title(main = "delta corr. MIAV", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_miav,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = methods_color[3],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+18], ")"), side = 3, adj = 0)
}
###############################################
corrplot(am_orig - am_hold, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. hldt", line = 1)
mtext(paste0("(", letters[26], ")"), side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = dat_hold,
                          leg_pos = leg_positions[j],
                          method_name = "holdout",
                          method_color = "green",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(z", j, ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)



##############################################################
## Generate Supplementary Figure with rho = -0.5 (Figure 7)
##############################################################

dat_orig <- dat_orig_0.5
dat_hold <- dat_hold_0.5
dat_synt_list <- list(syn_jf_0.5, syn_fc_0.5, syn_miav_0.5)
syn_jf <- dat_synt_list[[1]]
syn_fc <- dat_synt_list[[2]]
syn_miav <- dat_synt_list[[3]]


am_orig <- ComputeAssociationMatrix(dat_orig,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_hold <- ComputeAssociationMatrix(dat_hold,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_jf <- ComputeAssociationMatrix(syn_jf,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_fc <- ComputeAssociationMatrix(syn_fc,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_miav <- ComputeAssociationMatrix(syn_miav,
                                    num_variables = seq(5),
                                    cat_variables = NULL)


my_mar <- c(2.5, 2.5, 2, 0.5)
p <- 5

my_mar2 <- c(2, 0.2, 1.2, 0.2)

nms <- colnames(dat_orig)
par(mfrow = c(5, 6), mar = my_mar, mgp = c(1.5, 0.5, 0))
plot.new()
text(0.5, 0.5, expression(rho == -0.5), cex = 1.5)
corrplot(am_orig, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "original", line = 1)
mtext("(a)", side = 3, adj = 0)
corrplot(am_jf, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "JF", line = 1)
mtext("(b)", side = 3, adj = 0)
corrplot(am_fc, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "FC", line = 1)
mtext("(c)", side = 3, adj = 0)
corrplot(am_miav, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "MIAV", line = 1)
mtext("(d)", side = 3, adj = 0)
corrplot(am_hold, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "holdout", line = 1)
mtext("(e)", side = 3, adj = 0)
############################################
corrplot(am_orig - am_jf, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. JF", line = 1)
mtext("(f)", side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_jf,
                          leg_pos = leg_positions[j],
                          method_name = "JF",
                          method_color = methods_color[1],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+6], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_fc, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[12], ")"), side = 3, adj = 0)
title(main = "delta corr. FC", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_fc,
                          leg_pos = leg_positions[j],
                          method_name = "FC",
                          method_color = methods_color[2],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+12], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_miav, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[18], ")"), side = 3, adj = 0)
title(main = "delta corr. MIAV", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_miav,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = methods_color[3],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+18], ")"), side = 3, adj = 0)
}
###############################################
corrplot(am_orig - am_hold, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. hldt", line = 1)
mtext(paste0("(", letters[26], ")"), side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = dat_hold,
                          leg_pos = leg_positions[j],
                          method_name = "holdout",
                          method_color = "green",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(z", j, ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)


################################################################
## Generate Supplementary Figure with rho = -0.25 (Figure 8)
################################################################

dat_orig <- dat_orig_0.25
dat_hold <- dat_hold_0.25
dat_synt_list <- list(syn_jf_0.25, syn_fc_0.25, syn_miav_0.25)
syn_jf <- dat_synt_list[[1]]
syn_fc <- dat_synt_list[[2]]
syn_miav <- dat_synt_list[[3]]


am_orig <- ComputeAssociationMatrix(dat_orig,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_hold <- ComputeAssociationMatrix(dat_hold,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_jf <- ComputeAssociationMatrix(syn_jf,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_fc <- ComputeAssociationMatrix(syn_fc,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_miav <- ComputeAssociationMatrix(syn_miav,
                                    num_variables = seq(5),
                                    cat_variables = NULL)


my_mar <- c(2.5, 2.5, 2, 0.5)
p <- 5

my_mar2 <- c(2, 0.2, 1.2, 0.2)

nms <- colnames(dat_orig)
par(mfrow = c(5, 6), mar = my_mar, mgp = c(1.5, 0.5, 0))
plot.new()
text(0.5, 0.5, expression(rho == -0.25), cex = 1.5)
corrplot(am_orig, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "original", line = 1)
mtext("(a)", side = 3, adj = 0)
corrplot(am_jf, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "JF", line = 1)
mtext("(b)", side = 3, adj = 0)
corrplot(am_fc, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "FC", line = 1)
mtext("(c)", side = 3, adj = 0)
corrplot(am_miav, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "MIAV", line = 1)
mtext("(d)", side = 3, adj = 0)
corrplot(am_hold, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "holdout", line = 1)
mtext("(e)", side = 3, adj = 0)
############################################
corrplot(am_orig - am_jf, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. JF", line = 1)
mtext("(f)", side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_jf,
                          leg_pos = leg_positions[j],
                          method_name = "JF",
                          method_color = methods_color[1],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+6], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_fc, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[12], ")"), side = 3, adj = 0)
title(main = "delta corr. FC", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_fc,
                          leg_pos = leg_positions[j],
                          method_name = "FC",
                          method_color = methods_color[2],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+12], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_miav, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[18], ")"), side = 3, adj = 0)
title(main = "delta corr. MIAV", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_miav,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = methods_color[3],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+18], ")"), side = 3, adj = 0)
}
###############################################
corrplot(am_orig - am_hold, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. hldt", line = 1)
mtext(paste0("(", letters[26], ")"), side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = dat_hold,
                          leg_pos = leg_positions[j],
                          method_name = "holdout",
                          method_color = "green",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(z", j, ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)



##########################################################
## Generate Supplementary Figure with rho = 0 (Figure 9)
##########################################################

dat_orig <- dat_orig_0
dat_hold <- dat_hold_0
dat_synt_list <- list(syn_jf_0, syn_fc_0, syn_miav_0)
syn_jf <- dat_synt_list[[1]]
syn_fc <- dat_synt_list[[2]]
syn_miav <- dat_synt_list[[3]]

am_orig <- ComputeAssociationMatrix(dat_orig,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_hold <- ComputeAssociationMatrix(dat_hold,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_jf <- ComputeAssociationMatrix(syn_jf,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_fc <- ComputeAssociationMatrix(syn_fc,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_miav <- ComputeAssociationMatrix(syn_miav,
                                    num_variables = seq(5),
                                    cat_variables = NULL)

my_mar <- c(2.5, 2.5, 2, 0.5)
p <- 5

my_mar2 <- c(2, 0.2, 1.2, 0.2)

nms <- colnames(dat_orig)
par(mfrow = c(5, 6), mar = my_mar, mgp = c(1.5, 0.5, 0))
plot.new()
text(0.5, 0.5, expression(rho == 0), cex = 1.5)
corrplot(am_orig, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "original", line = 1)
mtext("(a)", side = 3, adj = 0)
corrplot(am_jf, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "JF", line = 1)
mtext("(b)", side = 3, adj = 0)
corrplot(am_fc, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "FC", line = 1)
mtext("(c)", side = 3, adj = 0)
corrplot(am_miav, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "MIAV", line = 1)
mtext("(d)", side = 3, adj = 0)
corrplot(am_hold, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "holdout", line = 1)
mtext("(e)", side = 3, adj = 0)
############################################
corrplot(am_orig - am_jf, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. JF", line = 1)
mtext("(f)", side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_jf,
                          leg_pos = leg_positions[j],
                          method_name = "JF",
                          method_color = methods_color[1],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+6], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_fc, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[12], ")"), side = 3, adj = 0)
title(main = "delta corr. FC", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_fc,
                          leg_pos = leg_positions[j],
                          method_name = "FC",
                          method_color = methods_color[2],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+12], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_miav, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[18], ")"), side = 3, adj = 0)
title(main = "delta corr. MIAV", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_miav,
                          leg_pos = leg_positions[j],
                          method_name = "MIAV",
                          method_color = methods_color[3],
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+18], ")"), side = 3, adj = 0)
}
###############################################
corrplot(am_orig - am_hold, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta corr. hldt", line = 1)
mtext(paste0("(", letters[26], ")"), side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = dat_hold,
                          leg_pos = leg_positions[j],
                          method_name = "holdout",
                          method_color = "green",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(z", j, ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)



######################################################
## Generate Figure 3 in the main text
######################################################

rho <- -0.75

set.seed(123)
X <- SimulateCorrelatedBetaData(n = n, 
                                rho = rho, 
                                beta_pars_list = beta_pars_list)
X[, 5] <- CategorizeVariable2(X[, 5], n_levels = 4)
set.seed(1234)
X[, 2] <- X[sample(n), 2]


M <- ComputeAuxiliaryVariables(X)
colnames(M) <- paste0("M", seq(5))

X2 <- X
X2[, 5] <- as.numeric(X2[,5])

am_X <- ComputeAssociationMatrix(X,
                                 num_variables = seq(4),
                                 cat_variables = 5)
am_M <- ComputeAssociationMatrix(M,
                                 num_variables = seq(4),
                                 cat_variables = 5)

leg_positions3 <- c("topright", "topright", "topleft", "bottom", "topright")

par(mfrow = c(1, 5), mar = c(2.75, 2.25, 1, 0.25) + 0.1, mgp = c(1.5, 0.5, 0))
par(mfrow = c(2, 6))
for (i in seq(5)) {
  MarginalDensityPlotsQC3(var_idx = i,
                          X = X2,
                          M = M,
                          leg_pos = leg_positions3[i],
                          main = bquote(M[.(i)] ~ "," ~ X[.(i)] ~ " distr."))
  mtext(paste0("(", letters[i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
corrplot(am_X, mar = c(1.5, 0, 1, 0), cl.pos = "n")
title(main = "assoc. of X", line = 0.4)
mtext(paste0("(", letters[6], ")"), side = 3, line = -1.2, at = 0, cex = 0.8)
####
for (i in seq(5)) {
  plot(X[, i], M[, i],
       xlab = bquote(italic(X[.(i)])),
       ylab = bquote(italic(M[.(i)])),
       main = bquote(M[.(i)] ~ " vs " ~ X[.(i)]),
       cex = 0.5)
  mtext(paste0("(", letters[6+i], ")"), side = 3, adj = 0, line = -1.2, cex = 0.8)
}
corrplot(am_M, mar = c(1.5, 0, 1, 0), cl.pos = "n")
title(main = "assoc. of M", line = 0.4)
mtext(paste0("(", letters[12], ")"), side = 3, line = -1.2, at = 0, cex = 0.8)
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, mgp = c(3, 1, 0))



######################################################
## Generate Supplemetary Figure comparing JF and 
## the updated-JF approaches (Figure 10)
######################################################

set.seed(123456789)
dat_orig <- SimulateCorrelatedBetaData(n = n, 
                                      rho = -0.75, 
                                      beta_pars_list = beta_pars_list)

dat_hold <- SimulateCorrelatedBetaData(n = n, 
                                      rho = -0.75, 
                                      beta_pars_list = beta_pars_list)



set.seed(12345)
syn_ujf1 <- UpdatedJointFactorizationTabPFNGenerator1(dat = dat_orig[sample(n),]) 
syn_ujf2 <- UpdatedJointFactorizationTabPFNGenerator2(dat = dat_orig[sample(n),]) 
syn_jf <- JointFactorizationTabPFNGenerator(X = dat_orig[sample(n),])


dat_orig <- dat_orig
dat_hold <- dat_hold
dat_synt_list <- list(syn_ujf1, syn_ujf2, syn_jf)
syn_ujf1 <- dat_synt_list[[1]]
syn_ujf2 <- dat_synt_list[[2]]
syn_jf <- dat_synt_list[[3]]


am_orig <- ComputeAssociationMatrix(dat_orig,
                                   num_variables = seq(5),
                                   cat_variables = NULL)
am_hold <- ComputeAssociationMatrix(dat_hold,
                                   num_variables = seq(5),
                                   cat_variables = NULL)
am_jf <- ComputeAssociationMatrix(syn_jf,
                                  num_variables = seq(5),
                                  cat_variables = NULL)
am_ujf1 <- ComputeAssociationMatrix(syn_ujf1,
                                    num_variables = seq(5),
                                    cat_variables = NULL)
am_ujf2 <- ComputeAssociationMatrix(syn_ujf2,
                                    num_variables = seq(5),
                                    cat_variables = NULL)


my_mar <- c(2.5, 2.5, 2, 0.5)
p <- 5

my_mar2 <- c(2, 0.2, 1.2, 0.2)

nms <- colnames(dat_orig)
par(mfrow = c(4, 6), mar = my_mar, mgp = c(1.5, 0.5, 0))
plot.new()
text(0.5, 0.5, expression(rho == -0.75), cex = 1.5)
corrplot(am_orig, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "original", line = 1)
mtext("(a)", side = 3, adj = 0)
corrplot(am_ujf1, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "UJF1", line = 1)
mtext("(b)", side = 3, adj = 0)
corrplot(am_ujf2, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "UJF2", line = 1)
mtext("(c)", side = 3, adj = 0)
corrplot(am_jf, mar = my_mar2, cl.pos = "n", col.lim = c(-1,1))
title(main = "JF", line = 1)
mtext("(d)", side = 3, adj = 0)
plot.new()
############################################
corrplot(am_orig - am_ujf1, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[5], ")"), side = 3, adj = 0)
title(main = "delta cor. UJF1", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_ujf1,
                          leg_pos = leg_positions[j],
                          method_name = "UJF1",
                          method_color = "purple",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+5], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_ujf1, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
title(main = "delta cor. UJF2", line = 1)
mtext(paste0("(", letters[11], ")"), side = 3, adj = 0)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_ujf2,
                          leg_pos = leg_positions[j],
                          method_name = "UJF2",
                          method_color = "cyan",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+11], ")"), side = 3, adj = 0)
}
############################################
corrplot(am_orig - am_jf, is.corr = FALSE, col.lim = c(-1,1), 
         cl.pos = "n", mar = my_mar2)
mtext(paste0("(", letters[17], ")"), side = 3, adj = 0)
title(main = "delta cor. JF", line = 1)
for (j in seq(p)) {
  MarginalDensityPlotsQC2(var_idx = j,
                          dat_real = dat_orig,
                          dat_synt = syn_jf,
                          leg_pos = leg_positions[j],
                          method_name = "JF",
                          method_color = "orange",
                          main = bquote(italic(X[.(j)])))
  mtext(paste0("(", letters[j+17], ")"), side = 3, adj = 0)
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)



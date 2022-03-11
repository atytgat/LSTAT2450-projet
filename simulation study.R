##################################################################
#                        SIMULATION STUDY                        #
##################################################################

# Packages and functions definitions
require(mvtnorm)
require(MLmetrics)
library(lars)
library(MASS)
library(glmnet)
library(caret)
library(flare)

# compute the SNR 
SNR <- function(f, epsilon) {
  var(f)/var(epsilon)
}

# performs one simulation 
sims = function(n, sd, Cov, p, beta, iseed){
  set.seed(iseed)
  eps = rnorm(n, 0, sd)
  x = rmvnorm(n,rep(0,p),Cov)
  y = x %*% beta + eps
  return (list("data" = (cbind(y, x)), "SNR" = SNR(y,eps)))
}

# Unchanged params :
nsim = 1000
p_lo = 10
p_hi = 150
sd_lo = 1
sd_hi = 10
n_lo = 30
n_hi = 200
rho_lo = 0.2
rho_hi = 0.7

# Choice of params : 
sd = sd_lo
n = n_lo
rho = rho_lo
p = p_lo

Cov = diag(rep(1, p)) # Covariance matrix
Cov[Cov==0] = rho
beta_true = rep(0,p) # true coefficients
beta_true[1:5] = c(0.4,7,3,3,3)
S_true = factor(rep('zero',p), levels = c('non-zero','zero'))  # active set
S_true[1:5] = 'non-zero'
S_true = factor(S_true)

### Simulation
# dataframe to record the results of each simulations
SNR.values = rep(0,nsim)
lm_results = setNames(as.data.frame(matrix(data = 0, nrow = nsim, ncol = 2)), 
                   c('acc_test','acc_beta'))
lm_coefs = as.data.frame(matrix(data = 0, nrow = nsim, ncol = p))


Ridge_results = setNames(as.data.frame(matrix(data = 0, nrow = nsim, ncol = 2)), 
                         c('acc_test','acc_beta'))
Ridge_coefs = as.data.frame(matrix(data = 0, nrow = nsim, ncol = p))


LASSO_results = setNames(as.data.frame(matrix(data = 0, nrow = nsim, ncol = 6)), 
                         c('acc_test','TPR','FPR','FNR','FDR','acc_beta'))
LASSO_coefs = as.data.frame(matrix(data = 0, nrow = nsim, ncol = p))

iseed = 123
for (i in 1:nsim) {
  results = sims(n, sd, Cov, p, beta_true, iseed+i)
  data = results$data
  SNR.values[i] = results$SNR
  
  # training and test set
  test = sample(1:n, n/2)
  train.set = as.data.frame(data[-test,])
  test.set = as.data.frame(data[test,])
  
  x.train = as.matrix(train.set[,2:(p+1)])
  y.train = as.matrix(train.set[,1])
  x.test =  as.matrix(test.set[,2:(p+1)])
  y.test = as.matrix(test.set[,1])
  
  # Ridge and LASSO parameters
  lbds_grid = seq(0,15, length = 100)
  if (n==n_lo) {
    nfold = 5
  } else {
    nfold = 10
  }
 
  # Unpenalized regression (if possible, else Ridge): 
  if (n/2>=p) {
    lm_fit = lm(V1~.-1, data = train.set)
    lm_pred = predict(lm_fit, test.set[,2:(p+1)])
    lm_results[i,] = c(MSE(lm_pred, test.set[,1]), MSE(beta_true, lm_fit$coefficients))
    lm_coefs[i,] = lm_fit$coefficients
    
  } else {
    Ridge_cv = cv.glmnet(x.train, y.train, lambda=lbds_grid, alpha=0, nfolds=nfold, intercept = FALSE)
    Ridge_pred = predict(Ridge_cv, s=Ridge_cv$lambda.min, newx=x.test)
    
    Ridge_results[i,] = c(MSE(Ridge_pred, y.test), MSE(as.matrix(coef(Ridge_cv))[-1], beta_true))
    Ridge_coefs[i,] = as.matrix(coef(Ridge_cv))[-1]
  }

  # LASSO regression
  LASSO_cv = cv.glmnet(x.train, y.train, lambda=lbds_grid, alpha=1, nfolds=nfold, intercept = FALSE)
  LASSO_pred = predict(LASSO_cv, s=LASSO_cv$lambda.min, newx=x.test)
  
  # estimated active set
  S_pred = as.matrix(coef(LASSO_cv))
  S_pred[S_pred!=0] = 'non-zero'
  S_pred[S_pred==0] = 'zero'
  S_pred = factor(S_pred, levels = c('non-zero','zero'))[2:(p+1)]
  
  LASSO_cm = confusionMatrix(S_pred, S_true)
  
  TPR = as.numeric(LASSO_cm$byClass[1]) 
  TNR = as.numeric(LASSO_cm$byClass[2])
  PPV = as.numeric(LASSO_cm$byClass[3])
  
  FPR = 1 - TNR 
  FNR = 1 - TPR 
  FDR = 1 - PPV 
  
  LASSO_results[i,] = c(MSE(LASSO_pred, y.test),
                        TPR, FPR, FNR, FDR,
                        MSE(as.matrix(coef(LASSO_cv))[-1], beta_true)
                        )
  LASSO_coefs[i,] = as.matrix(coef(LASSO_cv))[-1]
}




###############################################
#               TEST ZONE                     #
###############################################


c(round(mean(SNR.values),3),
  round(mean(lm_mat[,1]),3),
  round(mean(LASSO_mat[,1]),3),
  round(mean(LASSO_mat[,2]),3),
  round(mean(LASSO_mat[,3]),3),
  round(mean(LASSO_mat[,4]),3),
  round(mean(LASSO_mat[,5], na.rm=TRUE),3),
  round(mean(lm_mat[,2]),3),
  round(mean(LASSO_mat[,6]),3))

c(
  round(mean(SNR.values), 3),
  round(mean(Ridge_mat[, 1]), 3),
  round(mean(LASSO_mat[, 1]), 3),
  round(mean(LASSO_mat[, 2]), 3),
  round(mean(LASSO_mat[, 3]), 3),
  round(mean(LASSO_mat[, 4]), 3),
  round(mean(LASSO_mat[, 5], na.rm=TRUE), 3),
  round(mean(Ridge_mat[, 2]), 3),
  round(mean(LASSO_mat[, 6]), 3)
)


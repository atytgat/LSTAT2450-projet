# Using penalized models in statistical modeling

In this project, several estimation techniques, with special attention to penalized models, are applied in two different cases. 

The first one is a comparison study of simple linear regression vs. the Lasso method. This comparison is operated on a simulated dataset and the impacts of the following effects on the models are studied : low/high sample sizes, low/high correlation, low/high SNR and low/high dimensional data. In the settings where $p>n$ and a linear model can not be fitted, a Ridge regression is used instead. The performances are then evaluated on a test set by computing the MSE, the recovery of the active set and the MSE of the estimated coefficients. The results show that Lasso obtains better predictions than simple linear regression and Ridge in most cases and that it is always a simpler model (more sparse). The code can be found in the file 'simulation study.R'.

In the second part, variable selection procedures are used on a real life dataset to identify the relevant predictors of diabetes in a given population. Beforehand, a visualization analysis is presented to get an idea of the relevant predictors. Then, several models are fitted to the data such as  stepwise selection procedures, Elastic Nets, and a classification tree. Finally, the number of times each features were chosen is counted and the active set estimated is defined as the variables that were chosen by at least 60\% of the models. The results come close the initial expectations from the previous visualization analysis. The code can be found in the file 'variable selection.R'.

This project was conducted as part of the LSTAT2450 Statistical learning course at UCLouvain in the first term of 2020.

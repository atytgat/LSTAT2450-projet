##################################################################
#         Variable Selection on a Real-Life Dataset              #
##################################################################

# loading data and packages
data = read.table('C:/Users/alext/OneDrive - UCL/UCL/DATA/Q3/LSTAT2450 - Statistical learning. Estimation, selection and inference/RE _Exam_LSTAT2450/WHO-Diabetes.txt', header=T, sep=',')
library(AppliedPredictiveModeling)
library(MASS)
library(tree)
library(plyr)
library(glmnet)
library(ggplot2)
library(ggpubr)
transparentTheme(trans = .5)
library(caret)
data = data[complete.cases(data),] 
data$Diabetes = factor(data$Diabetes)

# training and test set creation
set.seed(123)
test = sample(1:nrow(data), 150)
train.set = data[-test,]
test.set = data[test,]


### visualization analysis
# correlations
featurePlot(x = data[, 1:8], 
            y = data$Diabetes, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))
# densities
featurePlot(x = data[, 1:8], 
            y = data$Diabetes, 
            plot = "density",
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4,2), 
            ## Add a key at the top
            auto.key = list(columns = 2))
# box plots
featurePlot(x = data[, 1:8], 
            y = data$Diabetes, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            ## Add a key at the top
            auto.key = list(columns = 2))


# balance of classes
ggplot(data, aes(Diabetes)) + geom_bar(fill = "#0073C2FF", width=0.5)
ggplot(train.set, aes(Diabetes)) + geom_bar(fill = "#0073C2FF", width=0.5)
ggplot(test.set, aes(Diabetes)) + geom_bar(fill = "#0073C2FF", width=0.5)



### Stepwise AIC/BIC logistic regression (forward=backward) 
# variable selection with septwise procedures
fit.int = glm(train.set$Diabetes~1, data=train.set, family=binomial())
fit.all = glm(train.set$Diabetes~., data=train.set, family=binomial())
mod.aic = stepAIC(fit.all, k=2, direction="backward", scope=list(upper=fit.all,lower=fit.int))
mod.bic = stepAIC(fit.all, k=log(nrow(data)), direction="backward", scope=list(upper=fit.all,lower=fit.int))
round(mod.aic$coefficients,3)
round(mod.bic$coefficients,3)

# fit the model with the variable selected
aic.fit = glm(mod.aic$formula, data=train.set, family=binomial())
bic.fit = glm(mod.bic$formula, data=train.set, family=binomial())
summary(aic.fit)

# prediction on the test set
prob.aic =  predict.glm(aic.fit, newdata = test.set[1:8], type = 'response') # probability for "pos"
pred.aic = rep("neg", length(prob.aic))
pred.aic[prob.aic > .5] = "pos"
round(mean(pred.aic==test.set$Diabetes),3)

prob.bic =  predict.glm(bic.fit, newdata = test.set[1:8], type = 'response') # probability for "pos"
pred.bic = rep("neg", length(prob.bic))
pred.bic[prob.bic > .5] = "pos"
round(mean(pred.bic==test.set$Diabetes),3)

### LASSO logistic :
lbd_grid = seq(0,20, length = 1000) # lambdas search space
X = as.matrix(train.set[,1:8])
Y = train.set$Diabetes

# cross validation to find the best lambda
cvlasso = cv.glmnet(x=X,y=Y,family="binomial", lambda = lbd_grid, alpha=1, nfolds = 10)
cvelnet = cv.glmnet(x=X,y=Y,family="binomial", lambda = lbd_grid, alpha=0.5, nfolds = 10)
round(coef(cvlasso),3)
round(coef(cvelnet),3)

# prediction on the test set
lasso.pred = predict(cvlasso, s=cvlasso$lambda.min, newx=as.matrix(test.set[,1:8]), type='class')
elnet.pred = predict(cvelnet, s=cvelnet$lambda.min, newx=as.matrix(test.set[,1:8]), type='class')
round(mean(lasso.pred==test.set$Diabetes),3)
round(mean(elnet.pred==test.set$Diabetes),3)

### classification tree 
# parameters of the tree + model estimation + plots
controls = tree.control(nrow(train.set), mincut = 5, minsize = 10, mindev = 0.01)
tree.fit = tree(train.set$Diabetes~.,train.set, split = 'deviance', control = controls)
plot(tree.fit,type="uniform")
text(tree.fit, cex=.75, digits=1)
# prediction on the test set
tree.pred = predict(tree.fit, test.set[1:8], type="class")
round(mean(tree.pred==test.set$Diabetes),3)
summary(tree.fit)

# prune the tree with cross validation to only keep the five best leaves of the tree
cv.fit=cv.tree(tree.fit,FUN=prune.tree)
prune.fit =prune.tree(tree.fit,best=5, method="misclass")
plot(cv.fit$size,cv.fit$dev,bty="n",pch=16)
plot(prune.fit,type="uniform")
text(prune.fit, cex=.85, digits=3)

# prediction on the test set
prune.pred = predict(prune.fit, test.set[1:8], type="class")
round(mean(prune.pred==test.set$Diabetes),3)
summary(prune.fit)


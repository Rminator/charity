####### Below is the R code used to Build the classification models and predictions models########
######### Below code builds models for charity data########################


library(Matrix)
library(gam)
library(MASS)
library(leaps) 
library(glmnet)
library(ggplot2)
library(tree) 
library(randomForest)
library(nnet)
library(gbm)
library(caret)
library(ggplot2)
library(pbkrtest)
library(glmnet)
library(lme4)
install.packages("verification")
require(verification)

# load the data
charity <- read.csv(file.choose()) # load the "charity.csv" file

#data set without the prediction and classification coloumn donr and damt

charitySub <- subset(charity,select = -c(donr,damt))

#Below function will check for missing data

sum(is.na(charitySub)) #There are no missing data among the other variables



# predictor transformations

charity.t <- charity
charity.t$avhv <- log(charity.t$avhv)

## check the predictor data to see if its skewed
dev.off()
par(mar=c(1,1,1,1))

hist(charity.t$avhv)
hist(chld)
hist(hinc)
hist(wrat)
hist(incm)
hist(inca)
hist(plow)
hist(npro)
hist(tgif)
hist(lgif)
hist(rgif)
hist(tdon)
hist(tlag)
hist(agif)


# set up data for analysis, partitioning the data into train/valid and test
## use part coloumn to seperate the data

data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:21]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)



######## The data looks pretty clean, so lets start building the models######
######### Below code will build a series of classification models ###########


##########Nearest neighbours###########

library(class)
 set.seed(1)
 post.valid.knn=knn(x.train.std,x.valid.std,c.train,k=5)
 profit.knn <- cumsum(14.5*c.valid[order(post.valid.knn, decreasing=T)]-2)
 n.mail.valid <- which.max(profit.knn)
 c(n.mail.valid, max(profit.knn))
 table(post.valid.knn,c.valid)
c(n.mail.valid, max(profit.knn))
# [1]  1247 11284
table(post.valid.knn,c.valid)
#c.valid  ## for k=15
#post.valid.knn   0   1
#0 713  49
#1 306 950
 1-mean(post.valid.knn==c.valid)
 plot(profit.knn)
 # 0.1759167
#check n.mail.valid = 306+950= 1256
# check profit = 14.5*950-2*1256 = 11263

# mailing and profits for different values of K
 #k=63
 #k=15
 #k=5
 #k=20
 #k=10
 

#################classification trees#####################
############### This gives around 84% accuracy############
 library(tree)
 attach(data.train.std.c)
 donr_class=ifelse(donr==1,"yes","no")
 data.train.std.c.tree=data.frame(data.train.std.c,donr_class)
 tree.model=tree(donr_class~.-donr,data.train.std.c.tree)
 
 #validation
 donr_class_valid=ifelse(data.valid.std.c$donr==1,"yes","no")
 data.valid.std.c.tree=data.frame(data.valid.std.c,donr_class_valid)
 summary(tree.model)
 
 #prediction
 model_pred=predict(tree.model,data.valid.std.c.tree,type="class")
 table(model_pred ,donr_class_valid)
 profit.tree <- cumsum(14.5*donr_class_valid[order(model_pred, decreasing=T)]-2)
 n.mail.valid <- which.max(profit.tree)
 c(n.mail.valid, max(profit.tree))
 table(model_pred,donr_class_valid)
 1-mean(model_pred==donr_class_valid)
 
 ## cross validation
 
cv.donr =cv.tree(tree.model,FUN=prune.misclass)
names(cv.donr)
cv.donr
#prune
prune.donr=prune.misclass(tree.model,best=5)
summary(prune.donr)
tree.pred=predict(prune.donr,data.valid.std.c.tree,type="class")
table(tree.pred ,donr_class_valid)



#####classification model: bagging ###############
##################################################

set.seed(1)
library(randomForest)
library(tree)
attach(data.train.std.c)
donr_class=ifelse(donr==1,"yes","no")
data.train.std.c.tree=data.frame(data.train.std.c,donr_class)
bag.model.class=randomForest(donr_class~.-donr,data=data.train.std.c.tree,mtry=10,importance=TRUE,ntree=800)

#11061.5

#validation
donr_class_valid=ifelse(data.valid.std.c$donr==1,"yes","no")
data.valid.std.c.tree=data.frame(data.valid.std.c,donr_class_valid)
summary(bag.model.class)
#prediction
set.seed(1)
model_pred_bag_class=predict(bag.model.class,data.valid.std.c.tree,type="class")
profit.bagging.tree <- cumsum(14.5*donr_class_valid[order(model_pred_bag_class, decreasing=T)]-2)
table(model_pred_bag_class,donr_class_valid)
1-mean(model_pred_bag_class==donr_class_valid)
plot(profit.bagging.tree)


############classification Random forest########
### k fold cross validation for  random forest

library(randomForest)
k=10
n=floor(nrow(data.train.std.c)/k)
err.vect=rep(NA,k)


ntrees=5000
for (i in 1:k){
  s1=((i-1) * n+1)
  s2=(i * n)
  subset=s1:s2
  
  cv.train=data.train.std.c[-subset,]
  cv.train=as.data.frame(data.train.std.c[-subset,])
  cv.test=as.data.frame(data.train.std.c[subset,])
  x=cv.train[,-21]
  y=cv.train[,21]
  fit=randomForest(x=cv.train[,-21],y=as.factor(cv.train[,21]))
  
  prediction=predict(fit,cv.test[,-21],type="prob")[,2]
  err.vect[i]=roc.area(cv.test[,21],prediction)$A
  print(paste("AUC for fold",i,":",err.vect[i]))
}

print(paste("Average AUC:",mean(err.vect)))



################### random forest model classification

set.seed(1)
library(randomForest)
library(tree)

data.train.std.c_bkp=as.data.frame(data.train.std.c)
rf.model.class=randomForest(donr_class~.-donr,data=data.train.std.c.tree,importance=TRUE)


model_pred_rf_class=predict(rf.model.class,data.valid.std.c.tree,type="class")
model_pred_rf_class=data.frame(model_pred_rf_class)
model_pred_rf_class=ifelse(model_pred_rf_class$model_pred_rf_class=="yes",1,0)
profit.random.forest <- cumsum(14.5*c.valid[order(model_pred_rf_class, decreasing=T)]-2)
n.mail.valid <- which.max(profit.random.forest)
table(model_pred_rf_class,c.valid)
1-mean(model_pred_rf_class==c.valid)
c(n.mail.valid, max(profit.random.forest))

donr_class_valid
#results:
#table(model_pred_rf_class,donr_class_valid)
#donr_class_valid
#model_pred_rf_class  no yes
#no  881  83
#yes 138 916
#> 1797/2018
#[1] 0.8904856


#################boosting models for classification

library(gbm)
set.seed(1)
fit.gbm1 <- gbm(donr~.,data=data.train.std.c,dist="adaboost",n.tree = 1000,shrinkage = 1)
gbm.perf(fit.gbm1)
confusion(predict(fit.gbm1, data.valid.std.c, n.trees = 400) > 0, data.valid.std.c$donr> 0)
confusion(predict(fit.gbm1, data.train.std.c,n.trees=26) > 0, data.train.std.c$donr > 0)


confusion <- function(a, b){
  tbl <- table(a, b)
  mis <- 1 - sum(diag(tbl))/sum(tbl)
  list(table = tbl, misclass.prob = mis)
}

########## gradient boosting ##########

library(gbm)
set.seed(1)
##t=35000
model.gboost <- gbm(donr~.,data= data.train.std.c,distribution = "bernoulli",n.trees=3500,interaction.depth=5,shrinkage = 0.01)
yhat.gboost<- predict(model.gboost,newdata=data.valid.std.c,n.trees=3500)
mean((yhat.gboost - data.valid.std.y)^2)


gboost.posterior.valid <- predict(model.gboost,n.trees=3500, data.valid.std.c, type="response") 
profit.gboost <- cumsum(14.5*c.valid[order(gboost.posterior.valid, decreasing=T)]-2)
n.mail.valid <- which.max(profit.gboost) 
c(n.mail.valid, max(profit.gboost)) 
cutoff.gboost <- sort(gboost.posterior.valid, decreasing=T)[n.mail.valid +1] 
gboost.valid <- ifelse(gboost.posterior.valid >cutoff.gboost, 1, 0) 
table(gboost.valid, c.valid)
1-mean(gboost.valid==c.valid)
plot(profit.gboost)



##########Model2##########
##########t=2500###########

set.seed(1)
model.2.gboost <- gbm(donr~.,data= data.train.std.c,distribution = "bernoulli",n.trees=2500,shrinkage=0.05,interaction.depth=4)
yhat.gboost.model2<- predict(model.2.gboost,newdata=data.valid.std.c,n.trees=2500)
mean((yhat.gboost.model2 - data.valid.std.y)^2)


gboost.posterior.valid.m2 <- predict(model.2.gboost,n.trees=2500, data.valid.std.c, type="response") 
profit.gboost <- cumsum(14.5*c.valid[order(gboost.posterior.valid.m2, decreasing=T)]-2)
n.mail.valid <- which.max(profit.gboost) 
c(n.mail.valid, max(profit.gboost)) 
cutoff.gboost <- sort(gboost.posterior.valid.m2, decreasing=T)[n.mail.valid +1] 
gboost.valid <- ifelse(gboost.posterior.valid.m2 >cutoff.gboost, 1, 0) 
table(gboost.valid, c.valid)
1-mean(gboost.valid==c.valid)
plot(profit.gboost)



############xgboost#######################

require(xgboost) 
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")

library(xgboost)
library(data.table)
library(readr)
library(stringr)
install.packages("data.table")


train<- data.train.std.c
test <- data.valid.std.c

setDT(train)
setDT(test)
labels <- train$donr
ts_label <- test$donr
new_tr <- model.matrix(~.+0,data = train[,-c("donr"),with=F])
new_ts <- model.matrix(~.+0,data = test[,-c("donr"),with=F])

labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

set.seed(1)
model.xgboost<- xgboost(data=new_tr,label=labels,nround=28,objective="binary:logistic",eval_metric="auc",
              early_stop_round = 10,max_depth=2)
pred<- predict(model.xgboost,new_ts)
head(pred)
profit.xgboost <- cumsum(14.5*c.valid[order(pred, decreasing=T)]-2)
n.mail.valid <- which.max(profit.xgboost)
c(n.mail.valid, max(profit.xgboost)) 

cutoff.xgboost <- sort(pred, decreasing=T)[n.mail.valid + 1] 
xgboost.valid <- ifelse(pred >cutoff.gboost, 1, 0) 
table(xgboost.valid, c.valid)

mean((pred - data.valid.std.y)^2)

plot(profit.xgboost)
cv.res<- xgb.cv(data=new_tr,nfold=10,label=labels,nround=50,objective="binary:logistic",eval_metric="auc")


####### cross validation for gbm########
library(gbm)
k=10
n=floor(nrow(data.train.std.c)/k)
err.vect=rep(NA,k)


ntrees=3500
for (i in 1:k){
  s1=((i-1) * n+1)
  s2=(i * n)
  subset=s1:s2
  
  cv.train=data.train.std.c[-subset,]
  cv.train=as.data.frame(data.train.std.c[-subset,])
  cv.test=as.data.frame(data.train.std.c[subset,])
  x=cv.train[,-21]
  y=cv.train[,21]

  fit= gbm.fit(x,y,n.trees=ntrees,verbose=FALSE,shrinkage=0.005,
              interaction.depth = 20,distribution="bernoulli")
  
  prediction=predict(fit,cv.test[,-21],n.trees=ntrees)
  err.vect[i]=roc.area(cv.test[,21],prediction)$A
  print(paste("AUC for fold",i,":",err.vect[i]))
}

print(paste("Average AUC:",mean(err.vect)))

### AUC values
#10--- 0.932625119237443
#1000-- 0.967426285396033
#2500-- 0.968249474094793
#3500-- 0.967796655536379


##### linear discriminant analysis

library(MASS)
model.lda1 <- lda(donr ~ reg1 +reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat +
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
                  data.train.std.c)

post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] 


profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1))

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table

1-mean(chat.valid.lda1==c.valid) 


##################



# logistic regression

logistic_model1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                         avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                       data.train.std.c, family=binomial("logit"))

#AIC 2189
logistic_model2 <- glm(donr ~ reg1 + reg2  + home + chld  + I(hinc^2)  + wrat + 
                         + incm + plow + npro + genf + tgif + tdon + tlag , 
                       data.train.std.c, family=binomial("logit"))

logistic_model3 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + 
                         hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + 
                         npro + tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c, family=binomial("logit"))



post.valid.logistic <- predict(logistic_model1,data.valid.std.c,type="response") # n.valid.c post probs
profit.logistic <- cumsum(14.5*c.valid[order(post.valid.logistic, decreasing=T)]-2)
plot(profit.logistic) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.logistic) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.logistic)) # report number of mailings and maximum profit
cutoff.logistic <- sort(post.valid.logistic, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.logistic <- ifelse(post.valid.logistic>cutoff.logistic, 1, 0) # mail to everyone above the cutoff
table(chat.valid.logistic, c.valid) # classification table
1-mean(chat.valid.logistic==c.valid)



#backward step elimination

step(lm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
        data=data.train.std.c),direction="backward")

logistic_model2 <- glm(donr ~ reg1 + reg2 + home + chld + hinc + I(hinc^2) + 
                         genf + wrat + incm + plow + npro + tgif + tdon + tlag,
                       data.train.std.c, family=binomial("logit"))




##forward direction profit 11642.5

step(lm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
        data=data.train.std.c),direction="forward")

logistic_model3 <- glm(donr ~ summaryreg1 + reg2 + reg3 + reg4 + home + chld + 
                         hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + 
                         npro + tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c, family=binomial("logit"))

post.valid.logistic <- predict(logistic_model3,data.valid.std.c,type="response") # n.valid.c post probs
profit.logistic <- cumsum(14.5*c.valid[order(post.valid.logistic, decreasing=T)]-2)
plot(profit.logistic) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.logistic) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.logistic)) # report number of mailings and maximum profit
cutoff.logistic <- sort(post.valid.logistic, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.logistic <- ifelse(post.valid.logistic>cutoff.logistic, 1, 0) # mail to everyone above the cutoff
table(chat.valid.logistic, c.valid) # classification table
1-mean(chat.valid.logistic==c.valid)


#both direction

step(lm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat +
          avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
        data=data.train.std.c),direction="both")

logistic_model4 <- glm(donr ~ reg1 + reg2 + home + chld + hinc + I(hinc^2) + 
                         genf + wrat + incm + plow + npro + tgif + tdon + tlag, data.train.std.c, family=binomial("logit"))


post.valid.logistic <- predict(logistic_model4,data.valid.std.c,type="response") # n.valid.c post probs
profit.logistic <- cumsum(14.5*c.valid[order(post.valid.logistic, decreasing=T)]-2)
plot(profit.logistic) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.logistic) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.logistic)) # report number of mailings and maximum profit
cutoff.logistic <- sort(post.valid.logistic, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.logistic <- ifelse(post.valid.logistic>cutoff.logistic, 1, 0) # mail to everyone above the cutoff
table(chat.valid.logistic, c.valid) # classification table
1-mean(chat.valid.logistic==c.valid)


############# QDA ##############

model.qda <- qda(donr ~ reg1 +reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat +
                   avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
                 data.train.std.c) # include additional terms on the fly using I()
post.valid.qda <- predict(model.qda, data.valid.std.c)$posterior[,2] # n.valid.c post probs
# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.qda <- cumsum(14.5*c.valid[order(post.valid.qda, decreasing=T)]-2)
plot(profit.qda) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda)) # report number of mailings and maximum profit
# 1418.0 11243.5
cutoff.qda <- sort(post.valid.qda, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda <- ifelse(post.valid.qda>cutoff.qda, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda, c.valid) # classification table
# c.valid
#chat.valid.qda 0 1
# 0 572 28
# 1 447 971
1-mean(chat.valid.qda==c.valid) #Error rate


################ SVM fit#################


set.seed(1)
data=data.frame(x=data.train.std.c[,-21],y=as.factor(data.train.std.c$donr))
library(e1071)
y=data.train.std.c$donr
svmfit=svm(y~.-data$x.donr, data=data, kernel="linear", cost=10,
           scale=FALSE)
tune=tune(svm,y~., data=data, kernel="linear", ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
bestmod=tune$best.model

validdata=data.frame(x=data.valid.std.c[,-21], y=as.factor(data.valid.std.c$donr))
ypred=predict(bestmod ,validdata)
table(predict=ypred, truth=validdata$y)
1-mean(ypred==c.valid)



###########################PART2#################################
################### regression/ predictive models################

#regression trees


reg.tree=tree(damt ~ .,data.train.std.y)
summary(reg.tree)
plot(reg.tree)
text(reg.tree,pretty=0)
cv.reg=cv.tree(reg.tree)
plot(cv.reg$size ,cv.reg$dev ,type='b')
yhat=predict(reg.tree ,newdata=data.valid.std.y)
mean((yhat-data.valid.std.y$damt)^2)
yhat.test.tree <- predict(reg.tree, newdata = data.test.std)
prune.reg=prune.tree(reg.tree ,best=11)
plot(prune.reg)
text(prune.reg,pretty=0)

yhat=predict(prune.reg ,newdata=data.valid.std.y)
mean((yhat-data.valid.std.y$damt)^2)
sd((y.valid - yhat)^2)/sqrt(n.valid.y)



################predictive model/regression: bagging and random forest###
#########################################################################


library(randomForest)
bag.model=randomForest(damt~.,data=data.train.std.y,mtry=20,importance=TRUE)
bag.model
summary(bag.model)
yhat.bag = predict(bag.model,newdata=data.valid.std.y)
mean((yhat.bag-data.valid.std.y$damt)^2)
sd((y.valid - yhat.bag)^2)/sqrt(n.valid.y)

yhat.test.bag <- predict(bag.model, newdata = data.test.std)
yhat.test
length(yhat.test)

############random forest
rf.model=randomForest(damt~.,data=data.train.std.y,importance=TRUE)
rf.model
yhat.rf=predict(rf.model,newdata=data.valid.std.y)
mean((yhat.rf-data.valid.std.y$damt)^2)
sd((y.valid - yhat.rf)^2)/sqrt(n.valid.y)
yhat.test.rf <- predict(rf.model, newdata = data.test.std)



#boosting
#gaussian distribution for regression and bernoulli distribution for the classification problem
#2000 trees
library(gbm)
boost.model=gbm(damt~.,data=data.train.std.y,distribution = "gaussian",n.trees=2000,interaction.depth =4,shrinkage=0.01)
yhat.boost=predict(boost.model,newdata=data.valid.std.y, n.trees=2000)
mean((yhat.boost-data.valid.std.y$damt)^2)
sd((y.valid - yhat.rf)^2)/sqrt(n.valid.y)
yhat.test.boost <- predict(boost.model, newdata = data.test.std,n.trees=2000)

########## 1500 trees

library(gbm)
boost.model.1=gbm(damt~.,data=data.train.std.y,distribution = "gaussian",n.trees=1500,interaction.depth =4,shrinkage=0.01)
yhat.boost=predict(boost.model,newdata=data.valid.std.y, n.trees=1500)
mean((yhat.boost-data.valid.std.y$damt)^2)
sd((y.valid - yhat.rf)^2)/sqrt(n.valid.y)
yhat.test.boost <- predict(boost.model, newdata = data.test.std,n.trees=1500)


#######################XGBOOST regression ##############################
library(xgboost)
require(xgboost) 
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
library(data.table)
library(readr)
library(stringr)
install.packages("data.table")


train<- data.train.std.y
test <- data.valid.std.y


setDT(train)
setDT(test)
labels <- train$damt
ts_label <- test$damt
new_tr <- model.matrix(~.+0,data = train[,-c("damt"),with=F])
new_ts <- model.matrix(~.+0,data = test[,-c("damt"),with=F])

labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1
dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

sparse<- xgb.cv(data = dtrain,nrounds = 600, min_child_weight = 0, max_depth = 10, eta = 0.02, 
      subsample = .7,colsample_bytree = .7, booster = "gbtree", eval_metric = "rmse", verbose = TRUE,print_every_n = 50, nfold = 4,nthread = 2, objective="reg:linear")


parameters <- list(colsample_bytree = .7, subsample = .7, booster = "gbtree", max_depth = 10, eta = 0.02,eval_metric = "rmse", objective="reg:linear")


Model3 <- xgb.train(params = parameters, data = dtrain, nrounds = 600, watchlist = list(train = dtrain), verbose = TRUE, print_every_n = 50, nthread = 2)

prediction <- predict(Model3, dtest)
prediction <- as.data.frame(as.matrix(prediction))
colnames(prediction) <- "prediction" 
finaloutput <- cbind(test, prediction)

mean((prediction-data.valid.std.y$damt)^2)
sd((y.valid - prediction)^2)/sqrt(n.valid.y)



#####################Multi regression model ##############

multireg_model1 <- lm(damt ~.,
                      data.train.std.y)

summary(multireg_model1)
pred.valid.ls1 <- predict(multireg_model1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y)


####stepwise
step(lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
        data=data.train.std.y),direction="forward")


step(lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat +
          avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,
        data=data.train.std.y),direction="both")

model_both<- lm (damt ~ reg3 + reg4 + home + chld + hinc + I(hinc^2) + 
                   genf + incm + plow + npro + rgif + tdon + agif, data = data.train.std.y)

summary(model_both)
pred.valid.ls1 <- predict(model_both, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error

sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y)


##############
multireg_model3 <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc  +
                        avhv + incm + plow + tgif + lgif + rgif  + tlag + agif,
                      data.train.std.y)

pred.valid.ls1 <- predict(multireg_model3, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y)


################### SVM  model for prediction######################


#Support Vector Machine (SVM)
library(e1071)
set.seed(1)
svm.model <- svm(damt ~.,kernel = "radial",data = data.train.std.y,cost=1,epsilon=0.2)
pred.valid.SVM.model <- predict(svm.model,newdata=data.valid.std.y)
mean((y.valid - pred.valid.SVM.model)^2) # mean prediction error
sd((y.valid - pred.valid.SVM.model)^2)/sqrt(n.valid.y) # std error

set.seed(1)
#10-fold cross validation for SVM using the default gamma of 0.5
# and using varying values of epsilon and cost
svm.tune <- tune(svm,damt~.,kernel = "radial",data=data.train.std.y,
                         ranges = list(epsilon = c(0.1,0.2,0.3,1.5,2), cost = c(0.01,1,5,10,50)))
summary(svm.tune)
#The SVM model has an epsilon of 0.2, a cost of 1 and a gamma of 0.5
svm.bst.model <- svm.tune$best.model
#For the SVM chosen; cost = 1, gamma =0.05 and epsilon=0.2
#There are 1,385 support vectors
summary(svm.bst.model$cost)
pred.valid.SVM.model1 <- predict(svm.bst.model,newdata=data.valid.std.y)
mean((y.valid - pred.valid.SVM.model1)^2) # mean prediction error
sd((y.valid - pred.valid.SVM.model1)^2)/sqrt(n.valid.y) # std error



################## RIDGE and LASSO###############


library(glmnet)
x=model.matrix(damt~.,data.train.std.y)
y=y.train
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
dim(coef(ridge.mod))
set.seed(1)
cv.out=cv.glmnet(x,y,alpha=0)
bestlam=cv.out$lambda.min
summary(bestlam)
valid.mm=model.matrix(damt~.,data.valid.std.y)
pred.valid.ridge=predict(ridge.mod,s=bestlam,newx=valid.mm)
mean((y.valid - pred.valid.ridge)^2) # mean prediction error
sd((y.valid - pred.valid.ridge)^2)/sqrt(n.valid.y) # std error


#######################Lasso########


lasso.mod=glmnet(x,y,alpha=1,lambda=grid)
set.seed(1)
cv.out=cv.glmnet(x,y,alpha=1)
cv.out
bestlam=cv.out$lambda.min
pred.valid.lasso=predict(lasso.mod,s=bestlam,newx=valid.mm)
mean((y.valid - pred.valid.lasso)^2) # mean prediction error
sd((y.valid - pred.valid.lasso)^2)/sqrt(n.valid.y) # std error


##########################################
##Principal Components Regression
#################


library(pls)
set.seed(1)
pcr.model=pcr(damt~.,data=data.train.std.y,scale=TRUE,validation="CV")
validationplot(pcr.model,val.type="MSEP")
pred.valid.pcr=predict(pcr.model,data.valid.std.y,ncomp=15)
mean((pred.valid.pcr-y.valid)^2)
sd((y.valid - pred.valid.pcr)^2)/sqrt(n.valid.y)


#################Classification submission of model##########

post.test <- predict(model.2.gboost,n.trees=2500, data.test.std, type="response") # post probs for testdata

# Oversampling adjustment for calculating number of mailings for test set
n.mail.valid <- which.max(profit.gboost)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set
cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)




################ Prediction submission of model ########
### i will be selecting GBM with 2000 trees and shrinkage = 0.01 (with Gaussian Distribution)

yhat.test <- predict(boost.model,n.trees = 2000, newdata = data.test.std) # test predictions


length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt


#########submission##########


ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="Nitin_Gaonkar.csv",
          row.names=FALSE) 








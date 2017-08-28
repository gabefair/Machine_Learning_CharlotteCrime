# Ryan's Working Directory
setwd("~/../Dropbox/Machine Learning Project/Data")
setwd("~/Dropbox/Machine Learning Project/Data/google_refine_cleaned_xlsx")
setwd("~/Dropbox/Machine Learning Project/Data")
#setwd("C:/Users/rwesslen/Dropbox/Machine Learning Project")
options(java.parameters = "-Xmx8192m")
library(rJava) 


Incident <- read.csv("./Dataset.csv", head=TRUE)


# dependent variable (Case_Status)
outcome<-"Clear_Flag"

Incident$Clear_Flag <- factor(Incident$Clear_Flag, levels = 0:1, labels = c("No", "Yes"))
Incident$DayofWeek <- factor(Incident$DayofWeek, labels = c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))
Incident$Year <- factor(Incident$Year)
Incident$ZipCode <- factor(Incident$ZipCode)
Incident$WithHeld <- factor(Incident$WithHeld)
Incident$NoAddByReported <- factor(Incident$NoAddByReported)

Incident$CSS_Called[Incident$CSS_Called == ""] <- "N"
Incident$CSS_Called <- factor(Incident$CSS_Called)


## Libraries
library(ggplot2)


# Training
library(caret)

#dependent variable
nzv <- nearZeroVar(Incident, saveMetrics= TRUE)
nzv[nzv$nzv,][1:10,]

inTrain<-createDataPartition(Incident$Clear_Flag, p = .7, list = FALSE)
train <-Incident[inTrain,]
test <-Incident[-inTrain,]



pos <- 1
predictors <-c("CSS_Called","Place1","Group","Against","WithHeld",
               "NoAddByReported","Place2","Category",
               "Location_Type","Long","Lat","Year","Week","DayofWeek","Hour")

#predictors <-c("Against")



catVars <- c("CSS_Called","Place1","Group","Against","WithHeld","NoAddByReported","Place2","Category",
             "Location_Type","Long","Lat","Year","DayofWeek")
numVars <- c("Long","Lat","Hour","Week")


# don't use "NIBRS_Hi_Class","Place2","Category","Division",
library(tabplot)

#exploratory analysis
tableplot(train[,c(outcome,predictors)])


# variable selection

logLikelyhood <-
      function(outCol,predCol) {
        sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
      }

selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(train[,outcome],sum(train[,outcome]==pos)/length(train[,outcome]))

for(v in catVars){
  pi <- paste('pred',v,sep='')
    liCheck <- 2*((logLikelyhood(train[,outcome],train[,pi]) - baseRateCheck))
    if(liCheck>minStep) {
        print(springf("%s, calibrationScore: %g",
          pi.liCheck))
        selVars <- c(selVars,pi)
    }
}

for(v in numVars){
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(train[,outcome],train[,pi]) - baseRateCheck) - 1)
if(liCheck>minStep) {
  print(springf("%s, calibrationScore: %g",
                pi.liCheck))
  selVars <- c(selVars,pi)
}
}

## non-h2o methods

# decision tree

library(rpart)				        # Popular decision tree algorithm
#library(rattle)					# Fancy tree plot
library(rpart.plot)				# Enhanced tree plots
library(RColorBrewer)				# Color selection for fancy tree plot
library(ebdbNet)
library(e1071)


fV <- paste(outcome, ' ~ ',paste(c(predictors),collapse=' + '),sep='')
tmodel <-rpart(fV,method = "class", data=train,
              control=rpart.control(cp=0.001,minsplit=1000,
              minbucket=1000,maxdepth=5))
print(tmodel)

plot(tmodel)					# Will make a mess of the plot
text(tmodel)

#fancyRpartPlot(tmodel)
#print(tmodel)

# test has (Accuracy : 0.8119, Sensitivity : 0.9041, Specificity : 0.6119)
trainhat <- predict(tmodel,newdata=train, type = "class")
confusionMatrix(trainhat, train$Clear_Flag)

# train has (Accuracy : 0.8147, Sensitivity : 0.9093, Specificity : 0.6095)
testhat <- predict(tmodel,newdata=test, type = "class")
confusionMatrix(testhat, test$Clear_Flag)

#print(calcAUC(predict(tmodel,newdata=train),train[,outcome]))
#print(calcAUC(predict(tmodel,newdata=test),test[,outcome]))

## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)

trainhat <- predict(tmodel,newdata=train, type = "prob")
testhat <- predict(tmodel,newdata=test, type = "prob")
testpred <- data.frame(testhat)
trainpred <- data.frame(trainhat)


## precision/recall curve (x-axis: recall, y-axis: precision)
perf1 <- performance(pred, "prec", "rec")
plot(perf1)
## sensitivity/specificity curve (x-axis: specificity,
## y-axis: sensitivity)
perf1 <- performance(pred, "sens", "spec")
plot(perf1)

knnpredictors <-c("Place1","Against",
               "NoAddByReported","Place2","Category",
               "Location_Desc","Year","Week","DayofWeek","Hour")

# k-nn
library(kknn)
knntrain <- train[c(knnpredictors,outcome)]
knntest <- test[c(knnpredictors,outcome)]

# test has (Accuracy : 0.7961, Sensitivity : 0.8850 , Specificity : 0.6034)
fit.kknn <- kknn(Clear_Flag ~ ., knntrain, knntest, k = 15, kernel = "triangular")
confusionMatrix(fit.kknn$fitted.values,test$Clear_Flag)

# 1 year takes about 10 minutes of run time
(fit.train1 <- train.kknn(y ~ ., knntrain, kmax = 15,
                          kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1))
table(predict(fit.train1, knntest), knntest$Clear_Flag)

#Type of response vari nominal
#Minimal misclassification: 
#Best kernel: triangular
#Best k: 15


# neural network
library(neuralnet)

# have to convert categorical variables into dummy variables
#dummies <- predict(dummyVars(~ NIBRS_Hi_Class, data = train), newdata = train)

#creditnet <- neuralnet(y ~ NIBRS_Hi_Class + Reporting_Agency + CSS_Called +
#                          Place1 + Place2 + Location_Type + Location_Desc + Division, 
#                       train, hidden = 3, lifesign = "minimal", 
#                       linear.output = FALSE, threshold = 0.1)




## h2o ------------------------------------------------------------------

#start h2o
suppressPackageStartupMessages(library(h2o))
localH20=h2o.init(nthreads = -1)
#h2o.checkClient(localH20)

# create two (training, test/validation) h2o datasets
train.hex <-as.h2o(localH20,train)
test.hex <-as.h2o(localH20,test)

predictors <-c("CSS_Called","Place1","Group","Against","WithHeld",
               "NoAddByReported","Place2","Category",
               "Location_Type","Long","Lat","Year","Week","DayofWeek","Hour","NIBRS_Hi_Class")


outcome <- "Clear_Flag"

summary(train.hex)

## K-Means --------------------------------------------------------------
# TBD

## Anamoly Detection ---------------------------------------------------
# See http://rpackages.ianhowson.com/cran/h2o/man/h2o.anomaly.html
train.dl = h2o.deeplearning(x = predictors, training_frame = train.hex, autoencoder = TRUE,
                              hidden = c(10, 10), epochs = 5)
train.anon = h2o.anomaly(train.dl, train.hex)
head(train.anon)



## Naive Bayes ----------------------------------------------------------
predictors <-c("Against","Place1","Location_Type","Category")

fitBayes<-h2o.naiveBayes(x = predictors, y = outcome, training_frame = train.hex)

# in sample test (Accuracy = 0.7557, Sensitivity : 0.8174, Specificity : 0.6219 )

bayes.train.pred <- as.data.frame(h2o.predict(fitBayes,train.hex,type="probs"))
bayes.train.CM <- confusionMatrix(bayes.train.pred$predict,train$Clear_Flag)

# out of sample test (Accuracy = 0.7586, Sensitivity : 0.8203, Specificity : 0.6249)

bayes.test.pred <- as.data.frame(h2o.predict(fitBayes,test.hex,type="probs"))
bayes.test.CM <- confusionMatrix(bayes.test.pred$predict,test$Clear_Flag)

## GLM (Logistic) --------------------------------------------------------
predictors <-c("CSS_Called","Place1","Group","Against","WithHeld",
               "NoAddByReported","Place2","Category",
               "Location_Type","Long","Lat","Year","Week","DayofWeek","Hour")

## Regularization: Lasso (alpha = 1)

fitGLM <-h2o.glm(x = predictors, y = outcome, training_frame = train.hex, family = "binomial", lambda_search = TRUE, alpha = 1)

# in sample test (Accuracy = 0.794 , Sensitivity : 0.8562 , Specificity : 0.6591)
GLM.train.pred <- as.data.frame(h2o.predict(fitGLM,train.hex,type="probs"))
GLM.train.CM <- confusionMatrix(GLM.train.pred$predict,train$Clear_Flag)

# out of sample test (Accuracy = 0.7955, Sensitivity : 0.8594, Specificity : 0.6569)
GLM.test.pred <- as.data.frame(h2o.predict(fitGLM,test.hex,type="probs"))
GLM.test.CM <- confusionMatrix(GLM.test.pred$predict,test$Clear_Flag)


## GBM ---------------------------------------------------------------------
predictors <-c("CSS_Called","Place1","Group","Against","WithHeld",
               "NoAddByReported","Place2","Category",
               "Location_Type","Long","Lat","Year","Week","DayofWeek","Hour")

fitgbm<-h2o.gbm(y=outcome,x=predictors, training_frame = train.hex,
                key="mygbm", distribution = "bernoulli", ntrees = 400,
                max_depth =5, interaction.depth = 2, learn_rate = 0.2)

# in sample test (Accuracy = 0.8537, Sensitivity : 0.8870, Specificity : 0.7814)
gbm.train.pred <- as.data.frame(h2o.predict(fitgbm,train.hex,type="probs"))
gbm.train.CM <- confusionMatrix(gbm.train.pred$predict,train$Clear_Flag)
confusionMatrix(gbm.train.pred$predict,train$Clear_Flag)

# out of sample test (Accuracy = 0.8171, Sensitivity : 0.8590 , Specificity : 0.7261)
gbm.test.pred <- as.data.frame(h2o.predict(fitgbm,test.hex,type="probs"))
gbm.test.CM <- confusionMatrix(gbm.test.pred$predict,test$Clear_Flag)
confusionMatrix(gbm.test.pred$predict,test$Clear_Flag)

gbm.varimp <- h2o.varimp(fitgbm)

## Deep Learning  ------------------------------------------------------------

fitDL <- h2o.deeplearning(x = predictors, y = outcome, training_frame = train.hex, 
                          hidden = c(200,200,200),variable_importances = TRUE)

# in sample test (Accuracy = 0.8148, Sensitivity: 0.8390, Specificity: 0.7625)
DL.train.pred <- as.data.frame(h2o.predict(fitDL, train.hex,type="probs"))
DL.train.CM <- confusionMatrix(DL.train.pred$predict,train$Clear_Flag)

# out of sample test (Accuracy = 0.8071, Sensivity: 0.8351, Specificity: 0.7462)
DL.test.pred <- as.data.frame(h2o.predict(fitDL,test.hex,type="probs"))
DL.test.CM <- confusionMatrix(DL.test.pred$predict,test$Clear_Flag)

dl.varimp <- h2o.varimp(fitDL)


### ROC Curves --------------------------------------------------------------

# CART Decision Tree
predbench <- prediction( trainpred$Yes, train$Clear_Flag)
trainbench <- performance(predbench,"tpr","fpr")
predbench <- prediction( testpred$Yes, test$Clear_Flag)
testbench <- performance(predbench,"tpr","fpr")

plot(perftest)
plot(perftrain, add = TRUE)
abline(a=0, b= 1)

#Need to convert predictions into probabilities

#Naive Bayes
predbayes <- prediction(bayes.train.pred$predict, train$Clear_Flag)
trainbayes <- performance(predbayes,"tpr","fpr")
predbayes <- prediction( bayes.test.pred$predict, test$Clear_Flag)
testbayes <- performance(predbayes,"tpr","fpr")


plot(trainbayes, add = TRUE)
plot(testbayes, add = TRUE)

#GLM
predglm <- prediction( GLM.train.pred$predict, train$Clear_Flag)
trainglm <- performance(predglm,"tpr","fpr")
predglm <- prediction( GLM.test.pred$predict, test$Clear_Flag)
testglm <- performance(predglm,"tpr","fpr")

plot(trainglm, add = TRUE)
plot(testglm, add = TRUE)

#GBM
predgbm <- prediction( GBM.train.pred$predict, train$Clear_Flag)
traingbm <- performance(predgbm,"tpr","fpr")
predgbm <- prediction( GBM.train.pred$predict, test$Clear_Flag)
testgbm <- performance(predgbm,"tpr","fpr")

plot(traingbm, add = TRUE)
plot(testgbm, add = TRUE)
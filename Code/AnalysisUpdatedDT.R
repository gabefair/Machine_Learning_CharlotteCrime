######MAKE SURE YOU HAVE THE LATEST R STUDIO########## (It now supports big data sized data.frames)


### Set Working Directory
setwd("C:\\Users\\Gabriel\\Dropbox\\Machine Learning Project\\Data")
#setwd("C:/Users/rwesslen/Dropbox/Machine Learning Project/Data")
#setwd("~/Dropbox/Machine Learning Project/Data")

### Initiate Libraries 
options(java.parameters = "-Xmx8192m")  #Increase memory -- you may want to ignore/comment out
library(rJava) 
library(ggplot2)
library(caret)
library(tabplot)
library(rpart)				        # Popular decision tree algorithm
library(rattle)					      # Fancy tree plot
library(rpart.plot)				    # Enhanced tree plots
library(RColorBrewer)			  	# Color selection for fancy tree plot
library(ebdbNet)
library(RWeka)
library(FSelector)              # Filter Variable Importance
library(e1071)                # SVM
library(spgwr)                # SWR (Spatially Weighted Regression)
library(kknn)                 # kNN
library(neuralnet)            # Neural Network
library(ROCR)
library(smbinning) 
library(ebdbNet)
library(maptools)


  ### Upload Dataset & Create Factors
  Incident <- read.csv("C:\\Users\\Gabriel\\Dropbox\\Machine Learning Project\\Data\\INCIDENT.csv", head=TRUE)
  
  Incident <- read.csv("INCIDENT.csv", head=TRUE)
  
  ## Use the "filter" below if you want to run on one type of Crime (e.g. Against == "Property" or "Person")
  Incident <- Incident[Incident$Against=="Property",]
  #Incident <- Incident[Incident$Against=="Person",]
  #Incident <- Incident[Incident$Against=="Other",]
  
  # dependent variable (Clear_Flag)
  outcome<-"Clear_Flag"
  
  # create factors for categorical and dummy variables
  Incident$Clear_Flag <- factor(Incident$ClearFlag, levels = 0:1, labels = c("No", "Yes"))
  #Incident$ClearFlag <- NULL
  Incident$YEAR <- factor(Incident$YEAR)
  Incident$ZipCode <- factor(Incident$ZipCode)
  
  Incident$Division <- factor(Incident$Division)
  Incident$RankPopDensity2010 <- factor(Incident$RankPopDensity2010)
  Incident$RankYouthPop2012 <- factor(Incident$RankYouthPop2012)
  Incident$RankWhitePop2010 <- factor(Incident$RankWhitePop2010)
  Incident$RankBachDeg2010 <- factor(Incident$RankBachDeg2010)
  Incident$RankVotPart2012 <- factor(Incident$RankVotPart2012)
  Incident$RankGrocProx2011 <- factor(Incident$RankGrocProx2011)
  Incident$RankTreeCanopy2012 <- factor(Incident$RankTreeCanopy2012)
  Incident$RankPublicAsst2010 <- factor(Incident$RankPublicAsst2010)
  Incident$RankAnimalControl2011 <- factor(Incident$RankAnimalControl2011)
  Incident$RankHHI2012 <- factor(Incident$RankHHI2012)
  Incident$RankHPI2012 <- factor(Incident$RankHPI2012)
  Incident$Black_Victim <- factor(Incident$Black_Victim)
  
  Incident$White_Victim <- factor(Incident$White_Victim)
  
  Incident$VULNERSUBST_FLAG <- factor(Incident$VULNERSUBST_FLAG)
  Incident$REFUSED_TREAT_FLAG <- factor(Incident$REFUSED_TREAT_FLAG)
  Incident$RankPropertyValue <- factor(Incident$RankPropertyValue)
  
  Incident$Within_Family_Victim_Flag <- factor(Incident$Within_Family_Victim_Flag)
  Incident$Outside_Family_Victim_Flag <- factor(Incident$Outside_Family_Victim_Flag)
  Incident$Unknown_Victim_Flag <- factor(Incident$Unknown_Victim_Flag)
  
  Incident$BUSINESS_FLAG <- factor(Incident$BUSINESS_FLAG)
  Incident$PUBLIC_FLAG <- factor(Incident$PUBLIC_FLAG)
  Incident$GOVT_FLAG <- factor(Incident$GOVT_FLAG)
  Incident$FIN_FLAG <- factor(Incident$FIN_FLAG)
  Incident$RELG_FLAG <- factor(Incident$RELG_FLAG)
  Incident$WALMART_FLAG <- factor(Incident$WALMART_FLAG)
  
  Incident$NCSTATE_FLAG <- factor(Incident$NCSTATE_FLAG)
  Incident$CHAR_FLAG <- factor(Incident$CHAR_FLAG)
  
  Incident$AddressSameFlag <- factor(Incident$AddressSameFlag)
  Incident$AddressReportFlag <- factor(Incident$AddressReportFlag)
  Incident$ReportByOfficerFlag <- factor(Incident$ReportByOfficerFlag)
  Incident$NameWithheldFlag <- factor(Incident$NameWithheldFlag)
  Incident$HomelessProxFlag <- factor(Incident$HomelessProxFlag)
  Incident$SchoolProxFlag <- factor(Incident$SchoolProxFlag)
  Incident$ChurchProxFlag <- factor(Incident$ChurchProxFlag)
  
  Incident$WinterWeatherFlag <- factor(Incident$WinterWeatherFlag)
  Incident$SevereWeatherFlag <- factor(Incident$SevereWeatherFlag)


## GGplot * Pre-Processing with Caret
# Check for near-zero-variance variables
nzv <- nearZeroVar(Incident, saveMetrics= TRUE)
nzv[nzv$nzv,][1:20,]

  # Create a 70 / 30 Training / Validation Partition
  set.seed(1234)
  inTrain <- createDataPartition(Incident$ClearFlag, p=0.7, list=FALSE) #inTrain has about 142,311 rows. 
  train <- Incident[inTrain,] #
  rest_of_set <- Incident[-inTrain,]

  # This will create another partition of the 30% of the data, so ~20%-testing and ~10%-validation
  inValidation <- createDataPartition(rest_of_set$ClearFlag, p=0.71, list=FALSE)
  valid <- rest_of_set[inValidation,]
  test <- rest_of_set[-inValidation,]

  # Choose which variables to use in the model as predictors
  predictors <-c("CSS_Called","Place1","NIBRS_Hi_Class","Against",
  "Place2","Category",
  "Location_Type","Long","Lat","YEAR","WEEK","DAYOFWEEK","HOUR",
  "RankPopDensity2010","RankYouthPop2012","RankWhitePop2010","RankBachDeg2010",
  "RankVotPart2012","RankGrocProx2011","RankTreeCanopy2012","RankPublicAsst2010",
  "RankAnimalControl2011","RankHHI2012","RankHPI2012","Black_Victim","White_Victim",
  "Victim_Age_Binned","VULNERSUBST_FLAG","REFUSED_TREAT_FLAG",
  "RankPropertyValue","Within_Family_Victim_Flag","Outside_Family_Victim_Flag","Unknown_Victim_Flag",
  "RollSevenDayNorm","BUSINESS_FLAG","WALMART_FLAG","NCSTATE_FLAG","CHAR_FLAG","PUBLIC_FLAG",
  "AddressSameFlag","AddressReportFlag","ReportByOfficerFlag","NameWithheldFlag",
  "HomelessProxFlag","SchoolProxFlag","ChurchProxFlag","WinterWeatherFlag","SevereWeatherFlag")


### Exploratory analysis with tabplot

#Crime Type
coltoview <-c("NIBRS_Hi_Class","Category","Against","Group")

tableplot(train[,c(outcome,coltoview)])

#Location
coltoview <- c("CSS_Called","Place1","Place2","Location_Type","Long","Lat","RollSevenDayNorm")

tableplot(train[,c(outcome,coltoview)])

#Location Ranks
coltoview <- c("RankPopDensity2010","RankYouthPop2012","RankWhitePop2010","RankBachDeg2010",
"RankVotPart2012","RankGrocProx2011","RankTreeCanopy2012","RankPublicAsst2010",
"RankAnimalControl2011","RankHHI2012")

tableplot(train[,c(outcome,coltoview)])


coltoview <- c("Black_Victim","White_Victim",
"Victim_Age","Victim_Age_Binned","VULNERSUBST_FLAG","REFUSED_TREAT_FLAG")

tableplot(train[,c(outcome,coltoview)])

coltoview <- c("RankPropertyValue","AddressSameFlag","AddressReportFlag","ReportByOfficerFlag","NameWithheldFlag",
"HomelessProxFlag","SchoolProxFlag","ChurchProxFlag")

tableplot(train[,c(outcome,coltoview)])

### variable selection (Filter / Model Independent Method)
#RocImp <- filterVarImp(x = train[, predictors], y = train$Clear_Flag, nonpara = FALSE)

#RocImp <- RocImp[with(RocImp, order(-Yes)), ]
#head(RocImp, top = 6)
#plot(RocImp$Yes, top = 20)




weights <- chi.squared(Clear_Flag~., train[-c(1,6,7,13)]) #Error : weka.core.UnsupportedAttributeTypeException: weka.filters.supervised.attribute.Discretize: Cannot handle multi-valued nominal class!
print(weights)
#subset <- cutoff.k(weights, 5)
#f <- as.simple.formula(subset, "Class")
#print(f)



#Temp <- Incident[c(20,22,39,43,44,45,46,70)]
#Temp$Clear_Flag <- as.numeric(Temp$Clear_Flag)
#Temp$Clear_Flag <- Temp$Clear_Flag - 1

# Binning
# http://blog.revolutionanalytics.com/2015/03/r-package-smbinning-optimal-binning-for-scoring-modeling.html
#VictimAge=smbinning(df=Temp,y="Clear_Flag",x="Victim_Age",p=0.05) 
#VictimAge$ivtable

# Set Victim Age breaks at 0 / Under 16 / 17 to 21 / 22 to 55 / 56 or Older




## non-h2o models

### simple CART

SIMPLEpredictors <- c("Against","PUBLIC_FLAG","RollSevenDayNorm","Victim_Age_Binned","Place1")
fV <- paste(outcome, ' ~ ',paste(c(SIMPLEpredictors),collapse=' + '),sep='')
simpletmodel <-rpart(fV,method = "class", data=train,control=rpart.control(cp=0.001,minsplit=1000,minbucket=1000,maxdepth=5))

# train has (Accuracy : 0.8313, Sensitivity : 0.9054, Specificity : 0.6705)
Simpletrainhat <- predict(simpletmodel,newdata=train, type = "class")
Simpledt.train.CM <- confusionMatrix(Simpletrainhat, train$Clear_Flag)
print(Simpledt.train.CM)

# valid has (Accuracy : 0.8312, Sensitivity : 0.9053, Specificity : 0.6703)
Simplevalidhat <- predict(simpletmodel,newdata=valid, type = "class")
Simpledt.valid.CM <- confusionMatrix(Simplevalidhat, valid$Clear_Flag)
print(Simpledt.valid.CM)

# test has (Accuracy : 0.8312, Sensitivity : 0.9053, Specificity : 0.6703)
Simpletesthat <- predict(simpletmodel,newdata=test, type = "class")
Simpledt.test.CM <- confusionMatrix(Simpletesthat, test$Clear_Flag)
print(Simpledt.test.CM)


### decision tree (CART)
#predictors <- c("Against","PUBLIC_FLAG","RollSevenDayNorm","Victim_Age_Binned","Place1")
fV <- paste(outcome, ' ~ ',paste(c(predictors),collapse=' + '),sep='')
tmodel <-rpart(fV,method = "class", data=train,control=rpart.control(cp=0.001,minsplit=1000,minbucket=1000,maxdepth=5))

print(tmodel)      # Print the decision rules
plot(tmodel)		# Will make a mess of the plot
text(tmodel)

fancyRpartPlot(tmodel) # Better visualization tool for decision trees
print(tmodel)

# train has (Accuracy : 0.8313, Sensitivity : 0.9054, Specificity : 0.6705)
trainhat <- predict(tmodel,newdata=train, type = "class")
dt.train.CM <- confusionMatrix(trainhat, train$Clear_Flag)
print(dt.train.CM)

# valid has (Accuracy : 0.8312, Sensitivity : 0.9053, Specificity : 0.6703)
validhat <- predict(tmodel,newdata=valid, type = "class")
dt.valid.CM <- confusionMatrix(validhat, valid$Clear_Flag)
print(dt.valid.CM)

# test has (Accuracy : 0.8312, Sensitivity : 0.9053, Specificity : 0.6703)
testhat <- predict(tmodel,newdata=test, type = "class")
dt.test.CM <- confusionMatrix(testhat, test$Clear_Flag)
print(dt.test.CM)



#Creating a blank vector for the model
tmodel_nl <- vector(mode="list", length=length(tmodel))
#save the names (Do not use)
#names(tmodel_nl) <- tmodel[-1]
#Convert tmodel to a vector
tmodel_v<- lapply(tmodel_nl, function(x) tmodel[1])

print(calcAUC(predict(tmodel,newdata=train, type = "class"),valid[,outcome])) #Not correct
print(calcAUC(predict(tmodel_v,newdata=valid),valid[,outcome])) #Not correct

#Setting up the prediction and performance class
data("C:\\Users\\Gabriel\\Dropbox\\Machine Learning Project\\Data\\INCIDENT.csv")
pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")

## precision/recall curve (x-axis: recall, y-axis: precision)
perf1 <- performance(pred, "prec", "rec")
plot(perf1)
## sensitivity/specificity curve (x-axis: specificity,
## y-axis: sensitivity)
perf1 <- performance(pred, "sens", "spec")
plot(perf1)


### Spatial GLM - see page 222-227, Ch 11 Spatial Data Analysis (TBD)
crime.coord <- cbind(train$Long,train$Lat)
crime.train <- SpatialPointsDataFrame(crime.coord, predictors, bbox = NULL)

# Calculate bandwidth caculation
gwr.model <- {Clear_Flag ~ Category + Place1}

set.bandwidth <- gwr.sel(gwr.model, data=train, verbose = FALSE, show.errormessages = FALSE)

#gwr.train.pred <- ggwr(gwr.model, bandwidth = set.bandwidth,
#                      predictions = TRUE, data = train, fit.points = crime.train, family = logistic)


### k-nn (TBD)
#knnpredictors <-c("Place1","Against",
#                  "NoAddByReported","Place2","Category",
#                  "Location_Desc","Year","Week","DayofWeek","Hour")

#knntrain <- train[c(knnpredictors,outcome)]
#knnvalid <- valid[c(knnpredictors,outcome)]

# valid has (Accuracy : 0.7961, Sensitivity : 0.8850 , Specificity : 0.6034)
#fit.kknn <- kknn(Clear_Flag ~ ., knntrain, knnvalid, k = 15, kernel = "triangular")
#confusionMatrix(fit.kknn$fitted.values,valid$Clear_Flag)

# 1 year takes about 10 minutes of run time
#(fit.train1 <- train.kknn(y ~ ., knntrain, kmax = 15,
#                          kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1))
#table(predict(fit.train1, knnvalid), knnvalid$Clear_Flag)



### neural network

# have to convert categorical variables into dummy variables
#dummies <- predict(dummyVars(~ NIBRS_Hi_Class, data = train), newdata = train)

#creditnet <- neuralnet(y ~ NIBRS_Hi_Class + Reporting_Agency + CSS_Called +
#                          Place1 + Place2 + Location_Type + Location_Desc + Division, 
#                       train, hidden = 3, lifesign = "minimal", 
#                       linear.output = FALSE, threshold = 0.1)


### SVM
# https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf
svm.model <- svm(fV, data = train, cost = 100, gamma = 1)
svm.pred <- predict(svm.model, valid)

### h2o ------------------------------------------------------------------

#start h2o
suppressPackageStartupMessages(library(h2o))
localH20=h2o.init(nthreads = -1)
#h2o.removeAll()		# only important if already had h2o session running
#h2o.checkClient(localH20)

# create Three (training, valid/validation) h2o datasets
train.hex <-as.h2o(localH20,train)
valid.hex <-as.h2o(localH20,valid)
test.hex <- as.h2o(localH20,test)

summary(train.hex)

## Anamoly Detection ---------------------------------------------------
# See http://rpackages.ianhowson.com/cran/h2o/man/h2o.anomaly.html
#train.dl = h2o.deeplearning(x = predictors, training_frame = train.hex, autoencoder = TRUE,
#                             hidden = c(10, 10), epochs = 5)
#train.anon = h2o.anomaly(train.dl, train.hex)
#head(train.anon)

## Naive Bayes ----------------------------------------------------------
predictorsNB <-c("Place2","Location_Type","Against","Category","AddressReportFlag","RankPropertyValue","PUBLIC_FLAG")

fitBayes<-h2o.naiveBayes(x = predictorsNB, y = outcome, training_frame = train.hex, laplace = 3)


# in sample train
bayes.train.pred <- as.data.frame(h2o.predict(fitBayes,train.hex,type="probs"))
bayes.train.CM <- confusionMatrix(bayes.train.pred$predict,train$Clear_Flag)
print(bayes.train.CM)

# out of sample valid (Accuracy = 0.7484, Sensitivity : 0.7719, Specificity : 0.6973)
bayes.valid.pred <- as.data.frame(h2o.predict(fitBayes,valid.hex,type="probs"))
bayes.valid.CM <- confusionMatrix(bayes.valid.pred$predict,valid$Clear_Flag)
print(bayes.valid.CM)

# Sample test (Accuracy = 0.7484, Sensitivity : 0.7719, Specificity : 0.6973)
bayes.test.pred <- as.data.frame(h2o.predict(fitBayes,test.hex,type="probs"))
bayes.test.CM <- confusionMatrix(bayes.test.pred$predict,test$Clear_Flag)
print(bayes.test.CM)




## GLM (Logistic) --------------------------------------------------------

## Regularization: Lasso (alpha = 1)

#Added the tree partitions. Fixed the glm parameters so it would run correctly.
fitGLM <-h2o.glm(x = predictors, y = outcome, training_frame = train.hex, family = "binomial", lambda_search = TRUE, alpha = 1)

# in sample train (Accuracy = 0.8175 , Sensitivity : 0.8827 , Specificity : 0.6758)
GLM.train.pred <- as.data.frame(h2o.predict(fitGLM,train.hex,type="probs"))
GLM.train.CM <- confusionMatrix(GLM.train.pred$predict,train$Clear_Flag)
print(GLM.train.CM)

# out of sample valid 
GLM.valid.pred <- as.data.frame(h2o.predict(fitGLM,valid.hex,type="probs"))
GLM.valid.CM <- confusionMatrix(GLM.valid.pred$predict,valid$Clear_Flag)
print(GLM.valid.CM)

# in sample test 
GLM.test.pred <- as.data.frame(h2o.predict(fitGLM,test.hex,type="probs"))
GLM.test.CM <- confusionMatrix(GLM.test.pred$predict,test$Clear_Flag)
print(GLM.test.CM)



#train
glm.varimp <- h2o.varimp(fitGLM)
print(glm.varimp)



## GBM ---------------------------------------------------------------------

fitgbm<-h2o.gbm(y=outcome,x=predictors, training_frame = train.hex,key="mygbm", distribution = "bernoulli", ntrees = 200,max_depth =5, interaction.depth = 2, learn_rate = 0.2)

# in sample valid (Accuracy = 0.8675, Sensitivity : 0.9020, Specificity : 0.7927)
gbm.train.pred <- as.data.frame(h2o.predict(fitgbm,train.hex,type="probs"))
gbm.train.CM <- confusionMatrix(gbm.train.pred$predict,train$Clear_Flag)
print(gbm.train.CM)

# out of sample valid (Accuracy = 0.8413, Sensitivity : 0.8851 , Specificity : 0.7464)
gbm.valid.pred <- as.data.frame(h2o.predict(fitgbm,valid.hex,type="probs"))
gbm.valid.CM <- confusionMatrix(gbm.valid.pred$predict,valid$Clear_Flag)
print(gbm.valid.CM)

# in sample test 
gbm.test.pred <- as.data.frame(h2o.predict(fitgbm,test.hex,type="probs"))
gbm.test.CM <- confusionMatrix(gbm.test.pred$predict,test$Clear_Flag)
print(gbm.test.CM)



gbm.varimp <- h2o.varimp(fitgbm)
print(gbm.varimp)  #<-- Look at the importance of crimes committed close to homeless shelters. 
# ^^^ But I think this is misleading as to homeless shelters just being located in areas of more crime.

## Deep Learning  ------------------------------------------------------------

fitDL <- h2o.deeplearning(x = predictors, y = outcome, training_frame = train.hex, hidden = c(200,200,200),variable_importances = TRUE)

# in sample valid (Accuracy = 0.8533, Sensitivity: 0.8929, Specificity: 0.7673)
DL.train.pred <- as.data.frame(h2o.predict(fitDL, train.hex,type="probs"))
DL.train.CM <- confusionMatrix(DL.train.pred$predict,train$Clear_Flag)
print(DL.train.CM)

# in valid sample
DL.valid.pred <- as.data.frame(h2o.predict(fitDL,valid.hex,type="probs"))
DL.valid.CM <- confusionMatrix(DL.valid.pred$predict,valid$Clear_Flag)
print(DL.valid.CM)

# in test sample
DL.test.pred <- as.data.frame(h2o.predict(fitDL,test.hex,type="probs"))
DL.test.CM <- confusionMatrix(DL.test.pred$predict,test$Clear_Flag)
print(DL.test.CM)


dl.varimp <- h2o.varimp(fitDL)
print(dl.varimp)

## Random Forests  ------------------------------------------------------------

fitRF <- h2o.randomForest(x = predictors, y = outcome, training_frame = train.hex, ntree = 50, max_depth = 10, min_rows = 5, nbins = 20)

# in sample valid (Accuracy = 0.8389, Sensitivity: 0.8654, Specificity: 0.7813)
RF.train.pred <- as.data.frame(h2o.predict(fitRF, train.hex,type="probs"))
RF.train.CM <- confusionMatrix(RF.train.pred$predict,train$Clear_Flag)
print(RF.train.CM)

# sample valid (Accuracy = 0.8302, Sensivity: 0.8606  , Specificity:  0.7643)
RF.valid.pred <- as.data.frame(h2o.predict(fitRF,valid.hex,type="probs"))
RF.valid.CM <- confusionMatrix(RF.valid.pred$predict,valid$Clear_Flag)
print(RF.valid.CM)

# sample test
RF.test.pred <- as.data.frame(h2o.predict(fitRF,test.hex,type="probs"))
RF.test.CM <- confusionMatrix(RF.test.pred$predict,test$Clear_Flag)
print(RF.test.CM)

rf.varimp <- h2o.varimp(fitRF)
print(rf.varimp)


### Test Only Scoring




### ROC Curves --------------------------------------------------------------

## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)

strainhat <- predict(simpletmodel,newdata=train, type = "prob")
svalidhat <- predict(simpletmodel,newdata=valid, type = "prob")
stesthat <- predict(simpletmodel,newdata=test, type="prob")

svalidpred <- data.frame(svalidhat)
strainpred <- data.frame(strainhat)
stestpred <- data.frame(stesthat)

# CART Decision Tree
predbench <- prediction( strainpred$Yes, train$Clear_Flag)
trainbench <- performance(predbench,"tpr","fpr")

predbench <- prediction( svalidpred$Yes, valid$Clear_Flag)
validbench <- performance(predbench,"tpr","fpr")

predbench <- prediction( stestpred$Yes, test$Clear_Flag)
testbench <- performance(predbench,"tpr","fpr")






strainROC <-roc(response=train$Clear_Flag,predictor=strainpred$Yes,levels=rev(levels(train$Clear_Flag)))
svalidROC <-roc(response=valid$Clear_Flag,predictor=svalidpred$Yes,levels=rev(levels(valid$Clear_Flag)))
stestROC <-roc(response=test$Clear_Flag,predictor=stestpred$Yes,levels=rev(levels(test$Clear_Flag)))


## complex CART

trainhat <- predict(tmodel,newdata=train, type = "prob")
validhat <- predict(tmodel,newdata=valid, type = "prob")
testhat <- predict(tmodel,newdata=test, type="prob")

validpred <- data.frame(validhat)
trainpred <- data.frame(trainhat)
testpred <- data.frame(testhat)

# CART Decision Tree
predcart <- prediction( trainpred$Yes, train$Clear_Flag)
traincart <- performance(predcart,"tpr","fpr")

predcart <- prediction( validpred$Yes, valid$Clear_Flag)
validcart <- performance(predcart,"tpr","fpr")

predcart <- prediction( testpred$Yes, test$Clear_Flag)
testcart <- performance(predcart,"tpr","fpr")




##get area under curve
library(pROC)

trainROC <-roc(response=train$Clear_Flag,predictor=trainpred$Yes,levels=rev(levels(train$Clear_Flag)))
validROC <-roc(response=valid$Clear_Flag,predictor=validpred$Yes,levels=rev(levels(valid$Clear_Flag)))
testROC <-roc(response=test$Clear_Flag,predictor=testpred$Yes,levels=rev(levels(test$Clear_Flag)))
plot(trainROC, type="S", print.thres= .5)
plot(validROC, type="S", print.thres= .5)
plot(testROC, type="S", print.thres= .5)

#

#Need to convert predictions into probabilities

library("ROCR")
#Naive Bayes
predbayes_train <- prediction(bayes.train.pred$Yes, train$Clear_Flag)
trainbayes <- performance(predbayes_train,"tpr","fpr")
trainbayes_AUC <- performance(predbayes_train, measure = "auc")
print(trainbayes_AUC@y.values)  #<--- This is the area under the curve

predbayes_valid <- prediction( bayes.valid.pred$Yes, valid$Clear_Flag)
validbayes <- performance(predbayes_valid,"tpr","fpr")
validbayes_AUC <- performance(predbayes_valid, measure = "auc")
print(validbayes_AUC@y.values)

predbayes_test <- prediction( bayes.test.pred$Yes, test$Clear_Flag)
testbayes <- performance(predbayes_test,"tpr","fpr")
testbayes_AUC <- performance(predbayes_test, measure = "auc")
print(testbayes_AUC@y.values)




#GLM
predglm_train <- prediction( GLM.train.pred$p1, train$Clear_Flag) #Variables come from code higer up in this file
trainglm <- performance(predglm_train,"tpr","fpr")
trainglm_AUC <- performance(predglm_train, measure = "auc") #Area under the ROC curve
print(trainglm_AUC@y.values)

predglm_valid <- prediction( GLM.valid.pred$p1, valid$Clear_Flag)
validglm <- performance(predglm_valid,"tpr","fpr")
validglm_AUC <- performance(predglm_valid, measure = "auc")
print(validglm_AUC@y.values)

predglm_test <- prediction( GLM.test.pred$p1, test$Clear_Flag)
testglm <- performance(predglm_test,"tpr","fpr")
testglm_AUC <- performance(predglm_test, measure = "auc")
print(testglm_AUC@y.values)




#GBM
predgbm_train <- prediction( gbm.train.pred$Yes, train$Clear_Flag)
traingbm <- performance(predgbm_train,"tpr","fpr")
traingbm_AUC <- performance(predgbm_train, measure = "auc")  #Area under the ROC
print(traingbm_AUC@y.values)

predgbm_valid <- prediction( gbm.valid.pred$Yes, valid$Clear_Flag)
validgbm <- performance(predgbm_valid,"tpr","fpr")
validgbm_AUC <- performance(predgbm_valid, measure = "auc")
print(validgbm_AUC@y.values)

predgbm_test <- prediction( gbm.test.pred$Yes, test$Clear_Flag)
testgbm <- performance(predgbm_test,"tpr","fpr")
testgbm_AUC <- performance(predgbm_test, measure = "auc")
print(testgbm_AUC@y.values)

#plot(traingbm, col = "brown", add = TRUE)


#DL
preddl_train <- prediction( DL.train.pred$Yes, train$Clear_Flag)
traindl <- performance(preddl_train,"tpr","fpr")
traindl_AUC <- performance(preddl_train, measure = "auc")
print(traindl_AUC@y.values)

preddl_valid <- prediction( DL.valid.pred$Yes, valid$Clear_Flag)
validdl <- performance(preddl_valid,"tpr","fpr")
validdl_AUC <- performance(preddl_valid, measure = "auc")
print(validdl_AUC@y.values)

preddl_test <- prediction( DL.test.pred$Yes, test$Clear_Flag)
testdl <- performance(preddl_test,"tpr","fpr")
testdl_AUC <- performance(preddl_test, measure = "auc")
print(testdl_AUC@y.values)

#plot(traindl, col = "blue", add = TRUE)


#RandomForest
predrf_train <- prediction( RF.train.pred$Yes, train$Clear_Flag)
trainrf <- performance(predrf_train,"tpr","fpr")
trainrf_AUC <- performance(predrf_train, measure = "auc")
print(trainrf_AUC@y.values)

predrf_valid <- prediction( RF.valid.pred$Yes, valid$Clear_Flag)
validrf <- performance(predrf_valid,"tpr","fpr")
validrf_AUC <- performance(predrf_valid, measure = "auc")
print(validrf_AUC@y.values)

predrf_test <- prediction( RF.test.pred$Yes, test$Clear_Flag)
testrf <- performance(predrf_test,"tpr","fpr")
testrf_AUC <- performance(predrf_test, measure = "auc")
print(testrf_AUC@y.values)

#plot(trainrf, col = "orange", add = TRUE)

#plot test 

plot(trainbench, col = "red", main="ROC by Each Model: Training Dataset",lwd=2.5)
abline(a=0, b= 1)
plot(traincart, col = "brown", add = TRUE,lwd=2.5)
plot(trainbayes, col = "green", add = TRUE,lwd=2.5)
plot(trainglm, col = "olivedrab2", add = TRUE,lwd=2.5)
plot(traingbm, col = "hotpink2", add = TRUE,lwd=2.5)
plot(traindl, col = "blue", add = TRUE,lwd=2.5)
plot(trainrf, col = "orange", add = TRUE,lwd=2.5)
legend("bottomright", # places a legend at the appropriate place 
       c("SimpleCART","CART","Naive Bayes","GLM","GBM","Deep Learning","Random Forest"), # puts text in the legend
       
       lty=c(1,1,1,1,1,1,1), # gives the legend appropriate symbols (lines)
       
       lwd=c(2.5,2.5,2.5,2.5,2.5,2.5,2.5),col=c("red","brown","green","olivedrab2","hotpink2","blue","orange"))

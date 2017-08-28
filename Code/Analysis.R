
setwd("~/Dropbox/Machine Learning Project/Data/google_refine_cleaned_xlsx")
options(java.parameters = "-Xmx8192m")
library(rJava) 

library(xlsx)
Incident2014 <- read.xlsx("Incident_Data_2014.xlsx",sheetIndex = 1)  
Incident2013 <- read.xlsx("Incident_Data_2013.xlsx",sheetIndex = 1)  
Incident2012 <- read.xlsx("Incident_Data_2012.xlsx",sheetIndex = 1)

## Change to the "cleanest" Incident dataset
Incident2014 <- read.csv("./Incident Data 2014.csv", head=TRUE)

Incident2014$Reported_Date <- strptime(x = as.character(Incident2014$Reported_Date),
                                format = "%m/%d/%Y %H:%M:%S")
Incident2014$Incident_To_Date <- strptime(x = as.character(Incident2014$Incident_To_Date),
                                       format = "%m/%d/%Y %H:%M:%S")
Incident2014$Incident_From_Date <- strptime(x = as.character(Incident2014$Incident_From_Date),
                                       format = "%m/%d/%Y %H:%M:%S")
Incident2014$Clearance_Date <- strptime(x = as.character(Incident2014$Clearance_Date),
                                            format = "%m/%d/%Y")

Incident2013 <- read.csv("./Incident Data 2013.csv")
Incident2013$Reported_Date <- strptime(x = as.character(Incident2013$Reported_Date),
                                       format = "%m/%d/%Y %H:%M:%S")
Incident2013$Incident_To_Date <- strptime(x = as.character(Incident2013$Incident_To_Date),
                                          format = "%m/%d/%Y %H:%M:%S")
Incident2013$Incident_From_Date <- strptime(x = as.character(Incident2013$Incident_From_Date),
                                            format = "%m/%d/%Y %H:%M:%S")
Incident2013$Clearance_Date <- strptime(x = as.character(Incident2013$Clearance_Date),
                                        format = "%m/%d/%Y")

Incident2012 <- read.csv("./Incident Data 2012.csv")
Incident2012$Reported_Date <- strptime(x = as.character(Incident2012$Reported_Date),
                                       format = "%m/%d/%Y %H:%M:%S")
Incident2012$Incident_To_Date <- strptime(x = as.character(Incident2012$Incident_To_Date),
                                          format = "%m/%d/%Y %H:%M:%S")
Incident2012$Incident_From_Date <- strptime(x = as.character(Incident2012$Incident_From_Date),
                                            format = "%m/%d/%Y %H:%M:%S")
Incident2012$Clearance_Date <- strptime(x = as.character(Incident2012$Clearance_Date),
                                        format = "%m/%d/%Y")

# dependent variable (Case_Status)
outcome<-"Case_Status"

# variables to keep
variables <-c("NIBRS_Hi_Class","CSS_Called","Place1","Place2",
              "Location_Type","Location_Desc","Division","Case_Status","Long","Lat","Reported_Date",
              "Clearance_Date")

## Need to fix
d <-rbind(Incident2014,Incident2013,Incident2012)
rm(Incident2014,Incident2013,Incident2012)
d <- d[,variables]

t.str <- strptime(d$Reported_Date, "%Y-%m-%d %H:%M:%S")

library(lubridate)
d$hour <- as.numeric(format(t.str, "%H"))
d$month <- as.numeric(format(t.str, "%m"))

temp <-c("NIBRS_Hi_Class","CSS_Called","Place1","Place2",
              "Location_Type","Location_Desc","Division","Case_Status","Long","Lat","hour","month")
d <- d[,temp]



## Libraries
library(ggplot2)

## Functions

# returns string w/o leading or trailing whitespace
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

## Variable Selection

# removes any records where Case_Status is null (1 in 2014)
d<-d[!(d$Case_Status=="null"),]

# keep only variables
data <- d[,temp]
rm(d)

data$Case_Status <- factor(trim(data$Case_Status))

data$y = factor(ifelse(data$Case_Status == "Close/Cleared",1,0))

data$Case_Status <- NULL


# Training
library(caret)

#dependent variable
#nzv <- nearZeroVar(data, saveMetrics= TRUE)
#nzv[nzv$nzv,][1:10,]

inTrain<-createDataPartition(data$y, p = .7, list = FALSE)
train <-data[inTrain,]
test <-data[-inTrain,]


catVars <- c("NIBRS_Hi_Class","CSS_Called","Place1","Place2",
             "Location_Type","Location_Desc","Division")
numVars <- c("Long","Lat","hour","month")

pos <- 1
predictors <-c("NIBRS_Hi_Class","CSS_Called","Place1","Place2",
         "Location_Type","Location_Desc","Division","Long","Lat","hour","month")

# 


#exploratory analysis
tableplot(d[,c(y,predictors)])

# Exploratory
ggplot(train) + geom_bar(aes(x=y,fill=month),position="fill")

summary(train)
month_tab <- data.frame(table(train$month,train$y))
hour_tab <- data.frame(table(train$hour,train$y))

# variable selection

logLikelyhood <-
      function(outCol,predCol) {
        sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
      }

selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(train[,y],sum(train[,y]==pos)/length(train[,y]))

for(v in catVars){
  pi <- paste('pred',v,sep='')
    liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) - baseRateCheck))
    if(liCheck>minStep) {
        print(springf("%s, calibrationScore: %g",
          pi.liCheck))
        selVars <- c(selVars,pi)
    }
}

for(v in numVars){
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) - baseRateCheck) - 1)
if(liCheck>minStep) {
  print(springf("%s, calibrationScore: %g",
                pi.liCheck))
  selVars <- c(selVars,pi)
}
}

## non-h2o methods

# decision tree

library(rpart)

fV <- paste(y, '>0 ~ ',paste(c(predictors),collapse=' + '),sep='')
tmodel <-rpart(fV,data=train,
              control=rpart.control(cp=0.001,minsplit=1000,
              minbucket=1000,maxdepth=5))

print(calcAUC)

# k-nn
library(kknn)

# test has (Accuracy : 0.743, Sensitivity : 0.8157, Specificity : 0.6467)
fit.kknn <- kknn(y ~ ., train,test, k = 15, kernel = "triangular")
xtab <- table(factor(fit.kknn$fitted.values),train$y)
confusionMatrix(xtab)



# 1 year takes about 10 minutes of run time
(fit.train1 <- train.kknn(y ~ ., train, kmax = 15,
                          kernel = c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1))
table(predict(fit.train1, test), test$y)

#Type of response vari nominal
#Minimal misclassification: 0.2419326
#Best kernel: triangular
#Best k: 15
library(ROCR)
pred <- prediction(predict(fit.train1, test), test$y)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(10))


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

predictors <-c("NIBRS_Hi_Class","CSS_Called","Place1","Place2",
               "Location_Type","Location_Desc","Division","Long","Lat","hour","month")
outcome <- "y"

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

fitBayes<-h2o.naiveBayes(x = predictors, y = outcome, training_frame = train.hex)

# in sample test (Accuracy = 0.5614, Sensitivity : 0.3333, Specificity : 0.8636 )
# (Add in Lat/Long, Hr/Month, Accuracy : 0.6022, Sensitivity : 0.4112, Specificity : 0.8553)
bayes.train.pred <- as.data.frame(h2o.predict(fitBayes,train.hex,type="probs"))
bayes.train.CM <- confusionMatrix(bayes.train.pred$predict,train$y)

# out of sample test (Accuracy = 0.5598, Sensitivity : 0.2926, Specificity : 0.8794)
# (Add in Lat/Long, Hr/Month, Accuracy : 0.6029, Sensitivity :0.4103, Specificity : 0.8580)
bayes.test.pred <- as.data.frame(h2o.predict(fitBayes,test.hex,type="probs"))
bayes.test.CM <- confusionMatrix(bayes.test.pred$predict,test$y)

## GLM (Logistic) --------------------------------------------------------

fitGLM <-h2o.glm(x = predictors, y = outcome, training_frame = train.hex, family = "binomial")

# in sample test (Accuracy = 0.7471, Sensitivity : 0.7032, Specificity : 0.8068)
GLM.train.pred <- as.data.frame(h2o.predict(fitGLM,train.hex,type="probs"))
GLM.train.CM <- confusionMatrix(GLM.train.pred$predict,train$y)

# out of sample test (Accuracy = 0.7430, Sensitivity : 0.7028, Specificity : 0.8062)
GLM.test.pred <- as.data.frame(h2o.predict(fitGLM,test.hex,type="probs"))
GLM.test.CM <- confusionMatrix(GLM.test.pred$predict,test$y)


## GBM ---------------------------------------------------------------------

fitgbm<-h2o.gbm(y=outcome,x=predictors, training_frame = train.hex,
                key="mygbm", distribution = "bernoulli", ntrees = 400,
                interaction.depth = 3)

# in sample test (Accuracy = 0.7677, Sensitivity : 0.7456, Specificity : 0.7970)
# (Add in Lat/Long, Hr/Month, Accuracy : 0.7817, Sensitivity :0.7641, Specificity : 0.8049)
gbm.train.pred <- as.data.frame(h2o.predict(fitgbm,train.hex,type="probs"))
gbm.train.CM <- confusionMatrix(gbm.train.pred$predict,train$y)

# out of sample test (Accuracy = 0.7608, Sensitivity : 0.7405, Specificity : 0.7877)
# (Add in Lat/Long, Hr/Month, Accuracy : 0.7646, Sensitivity :0.7460, Specificity : 0.7892)
gbm.test.pred <- as.data.frame(h2o.predict(fitgbm,test.hex,type="probs"))
gbm.test.CM <- confusionMatrix(gbm.test.pred$predict,test$y)


## Deep Learning  ------------------------------------------------------------

fitDL <- h2o.deeplearning(x = predictors, y = outcome, training_frame = train.hex)

# in sample test (Accuracy = 0.7644)
DL.train.pred <- as.data.frame(h2o.predict(fitDL, train.hex,type="probs"))
DL.train.CM <- confusionMatrix(DL.train.pred$predict,train$y)

# out of sample test (Accuracy = 0.7593)
DL.test.pred <- as.data.frame(h2o.predict(fitDL,test.hex,type="probs"))
DL.test.CM <- confusionMatrix(DL.test.pred$predict,test$y)



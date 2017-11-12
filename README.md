---
title: "Practical Machine Learning - Human Activity Recognition"
author: "Raed Abdel Sater"
date: "November 12, 2017"
output:
  html_document: default
  pdf_document: default
  
---


Human Activity Recognition - HAR - has emerged as a key research area in the last years and is gaining increasing attention by the pervasive computing research community (see picture below, that illustrates the increasing number of publications in HAR with wearable accelerometers), especially for the development of context-aware systems. There are many potential applications for HAR, like: elderly monitoring, life log systems for monitoring energy expenditure and for supporting weight-loss programs, and digital assistants for weight lifting exercises.

In this project, data was used from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

First we prepare the packages needed to load and manipulate data. along side to Caret and randomForcast, I used Hmisc to analyze data and doParallel package to decrease the random forest processing time by running parallel execution processing on Training data set. The seed value was set at the beginning of this project to garantee reproducible results



````r
options(warn=-1)
library(caret)
library(randomForest)
library(Hmisc)
library(foreach)
library(doParallel)
```

```r
set.seed(4356)
```

The first step is to load the csv file data to dataframe and analyze the type & the completion rate of the data (commands are commented to limit the output size. You can run it deleting the "#" ) :


```r
setwd("/home/jagoul/Coursera/Data-Science-Specialization/Ptractical Machine Learning/Projects/Final Project/")
if (!file.exists("pmlTraining.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                  destfile = "pmlTraining.csv")
}

if (!file.exists("pmlTesting.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                  destfile = "pmlTesting.csv")
}
data <- read.csv("pmlTraining.csv")

#summary(data)
#describe(data)
#sapply(data, class)
#str(data)
```

After observing the data we can deduce that some numeric data have been imported as factor,the presence of characters such as ("#DIV/0!") explain this observation. Also the fact that some columns have a really low completion rate, and consequently a lot of missing data values are present inside the data set.
 
To manage the factor issue,  we impute data while ignoring "#DIV/0!" values as follows:

```r
data <- read.csv("/projects/Coursera-PracticalMachineLearning/data//pml-training.csv", na.strings=c("#DIV/0!") )
```

Then run a numeric casting on all affected columns 


```r
cData <- data
for(i in c(8:ncol(cData)-1)) {cData[,i] = as.numeric(as.character(cData[,i]))}
```

To manage the second issue we will select as feature only the column with a 100% completion rate ( as seen in analysis phase, the completion rate in this dataset is very binary) We will also filter some features which seem to be useless like "X"", timestamps, "new_window" and "num_window". We filter also user_name because we don't want learn from this feature (name cannot be a good feature in our case and we don't want to limit the classifier to the name existing in our training dataset)


```r
featuresnames <- colnames(cData[colSums(is.na(cData)) == 0])[-(1:7)]
features <- cData[featuresnames]
```


We have now a dataframe "features which contains all the workable features. So the first step is to split the dataset in two part : the first for training and the second for testing.


```r
xdata <- createDataPartition(y=features$classe, p=3/4, list=FALSE )
training <- features[xdata,]
testing <- features[-xdata,]
```


We can now train a classifier with the training data. To do that we will use parallelise the processing with the foreach and doParallel package : we call registerDoParallel to instantiate the configuration. (By default it's assign the half of the core available on your laptop, for me it's 4, because of hyperthreading) So we ask to process 4 random forest with 150 trees each and combine then to have a random forest model with a total of 600 trees.

```r
registerDoParallel()
model <- foreach(ntree=rep(150, 4), .combine=randomForest::combine) %dopar% randomForest(training[-ncol(training)], training$classe, ntree=ntree)
```

To evaluate the model we will use the confusionmatrix method and we will focus on accuracy, sensitivity & specificity metrics :

```r
predictionsTr <- predict(model, newdata=training)
confusionMatrix(predictionsTr,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
predictionsTe <- predict(model, newdata=testing)
confusionMatrix(predictionsTe,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  946    6    0    0
##          C    0    2  849    6    1
##          D    0    0    0  798    1
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.993    0.993    0.998
## Specificity             1.000    0.998    0.998    1.000    1.000
## Pos Pred Value          0.999    0.994    0.990    0.999    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.183
## Detection Prevalence    0.285    0.194    0.175    0.163    0.183
## Balanced Accuracy       1.000    0.998    0.995    0.996    0.999
```

The result of the confusionmatrix now is ready, as we can see, the model is performance is quite good and efficient with an an accuracy of 0.997 and very good sensitivity & specificity values on the testing dataset. The lowest value is 0.993 for the sensitivity of the class C)


The results are quite impressive, the 20 predictable values has a score of 100% with 20 / 20 hits.
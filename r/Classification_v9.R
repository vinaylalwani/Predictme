


# **************************************************************************** #
#                 READ THE DATA
# **************************************************************************** #
require(data.table)
orig_w1_data_old=fread('/Users/vinaylalwani/per/projects/ml/SR/W1.csv')
orig_w12_data_old=fread('/Users/vinaylalwani/per/projects/ml/SR/W12.csv')
# filter records where Sales ID is not [N/A, empty] and State is unspecified and remove 'Creator ID' from the table
w1_data_old = orig_w1_data_old[`Sales ID` != '#N/A' & `State` != 'Unspecified'][order(`Sales ID`, `Customer ID`)][, c("Creator ID"):=NULL]
w1_data_old = w1_data_old[, ("Sales ID") := lapply(.SD, as.numeric), .SDcols = "Sales ID"]

orig_w1_data=fread('/Users/vinaylalwani/per/projects/ml/SR/v9.0_W1.csv')
orig_w12_data=fread('/Users/vinaylalwani/per/projects/ml/SR/v9.0_W12.csv')
# filter records where Sales ID is not [N/A, empty] and State is unspecified, and remove 'Opportunity ID' and 'Opportunity Created by' fields
w1_data = orig_w1_data[`Sales ID` != '#N/A' & `Sales ID` != '' & `State` != 'Unspecified'][order(`Sales ID`)][, c("Opportunity Created by"):=NULL]
# convert sales id into numeric
w1_data = w1_data[, ("Sales ID") := lapply(.SD, as.numeric), .SDcols = "Sales ID"]


# **************************************************************************** #
#                 PRE PROCESS THE DATA
# **************************************************************************** #

# Encoding categorical data
# uncomment this if country is present in the data
# w1_data$Country = factor(w1_data$Country, levels = c("United States", "Canada"), labels = c(1, 0))
w1_data$State = factor(w1_data$State, 
                       levels = c("CA","NJ","OH","IL","AR","MD","FL","MA","GA","WA","NY","MI","IN","AZ","CO","TX","NB","ON","VA","NC","QC","PA","AL","BC","SC","KS","DC","OK","OR","LA","CT","AB","MO","WV","NE","TN","RI","NM","NV","MN","SK","ME","WI","VT","NH","KY","MB","IA","ID","DE","ND","HI","UT","MT"), 
                       labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54))

# w12_data = orig_w12_data[`Sales ID` != '#N/A' & `Sales ID` != '' & `State` != 'Unspecified'][order(`Sales ID`, `Customer ID`)][, c("Opportunity Created by"):=NULL]
w12_data = orig_w12_data[`Sales ID` != '#N/A' & `Sales ID` != '' & `State` != 'Unspecified' & `Opportunity ID` != '#N/A'][order(`Sales ID`)][, c("Opportunity Created by"):=NULL]
# convert sales id into numeric
w12_data = w12_data[, ("Sales ID") := lapply(.SD, as.numeric), .SDcols = "Sales ID"]
w12_data = w12_data[, ("Opportunity ID") := lapply(.SD, as.numeric), .SDcols = "Opportunity ID"]
# uncomment this if country is present in the data
# w12_data$Country = factor(w12_data$Country, levels = c("United States", "Canada"), labels = c(1, 0))
w12_data$State = factor(w12_data$State, 
                        levels = c("CA","NJ","OH","IL","AR","MD","FL","MA","GA","WA","NY","MI","IN","AZ","CO","TX","NB","ON","VA","NC","QC","PA","AL","BC","SC","KS","DC","OK","OR","LA","CT","AB","MO","WV","NE","TN","RI","NM","NV","MN","SK","ME","WI","VT","NH","KY","MB","IA","ID","DE","ND","HI","UT","MT"), 
                        labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54))

# intersection of w1 and w12 only
# successData = merge(w1_data, w12_data)[order(`Sales ID`, `Customer ID`)]
successData = merge(w1_data, w12_data)[order(`Sales ID`)]
successData_1<-successData[!(successData$`Input Probability` >= 50 ),]
# mergeData = merge(w1_data, w12_data, by = c("Opportunity ID"), all = TRUE)
# mergeData = merge(w1_data, w12_data, 
#                  by = c("Opportunity ID"), all = TRUE)[, c("Country.x", "State.x", "Customer ID.y", "Sales ID.y", "Country.y", "State.y"):=NULL]
mergeData = merge(w1_data, w12_data, 
                  by = c("Opportunity ID"), all = TRUE)[, c("State.x", "Sales ID.y", "State.y"):=NULL]

# get the number transactions of a given sales id
data.frame(table(mergeData$`Sales ID.x`))
# get the number transactions of a given customer id
#data.frame(table(mergeData$`Customer ID.x`))

# convert all NA to zeroes to prepare the training set
mergeData[is.na(mergeData)] <- 0
# choose a sales id to run the algorithm; working: 10053, 10062; not working: 10032; tbd: 10043
sampleDataSet = mergeData[`Sales ID.x`==10062]
# select specific fields
sampleDataSet = sampleDataSet[, c(FALSE,FALSE,TRUE,TRUE,TRUE,FALSE)]
# just a buffer, for verification only.
origSampleDataSet = sampleDataSet
# remove rows whose input probability < 60%
sampleDataSet<-sampleDataSet[!(sampleDataSet$`Input Probability` >= 60 ),]
# probability in decimal format
sampleDataSet$`Input Probability` <- sampleDataSet$`Input Probability` / 100
sampleDataSet$`Output Probability`[sampleDataSet$`Output Probability` == 100] <- 1.0

# **************************************************************************** #
#     TRAINING AND TEST SET
# **************************************************************************** #
# prepare the training and test dataset
library(caTools)
set.seed(123)
# make a 80-20 split
split = sample.split(sampleDataSet$`Output Probability`, SplitRatio = 0.8)
training_set = subset(sampleDataSet, split == TRUE)
test_set = subset(sampleDataSet, split == FALSE)
# **************************************************************************** #

# **************************************************************************** #
# Feature scaling
# **************************************************************************** #
#training_set[, 1:1] = scale(training_set[, 1:1])
#test_set[, 1:1] = scale(test_set[, 1:1])
# **************************************************************************** #

# %%%%%%%%%%%% Processing begins here %%%%%%%%%%%%%

################################################################################
# fit linear regression to the training set
################################################################################
#regressor = lm(formula = `Output Probability` ~ `Input Probability` + `Opportunity ID` + `Customer ID.x` + `Predicted Revenue`,
#               data = training_set)
regressor = lm(formula = `Output Probability` ~ `Input Probability` + `Customer ID.x` ,
               data = training_set)

summary(regressor)
# Plotting the tree
plot(regressor)
text(regressor)

# predicting the Test set results
y_pred = predict(regressor, newdata = test_set[, -3])
#y_pred = ifelse(y_pred > 0.8, 1, 0)
y_pred

# Visualising the Training set results : Logistic regression
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Customer ID.x', 'Input Probability')
prob_set = predict(regressor, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.8, 1, 0)
plot(set[, -3],
     main = 'Linear Regression (Training set)',
     xlab = 'Customer ID.x', ylab = 'Input Probability', 
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
################################################################################

# CLASSIFICATION ALGORITHMS #
# ------------------------- #

################################################################################
# Fitting Logistic Regression to the Training set
################################################################################
classifier = glm(formula = `Output Probability` ~ .,
                 family = binomial,
                 data = training_set)
# predicting the Test set results
y_pred = predict(classifier, type = 'response', newdata = test_set[, -3])
y_pred = ifelse(y_pred > 0.8, 1, 0)
y_pred

# Making the Confusion Matrix
table(unlist(test_set[, 3]), y_pred)

summary(classifier)
# Plotting the tree
plot(classifier)
text(classifier)

# Visualising the Training set results : Logistic regression : TBD:
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by = 0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Customer ID.x', 'Input Probability')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.8, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Customer ID.x', ylab = 'Input Probability', 
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
################################################################################

################################################################################
# Fitting K-NN to the training set and predicting the test set results
################################################################################
library(class)
y_pred = knn(train = training_set[, -3],
             test = test_set[, -3],
             cl = training_set[, 3],
             k = 5)
################################################################################

################################################################################
# SVM: Fitting classifier Regression to the Training set : SVM for classification and SVR for regression
################################################################################
library(e1071)
classifier = svm(formula = `Output Probability` ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# predicting the Test set results
y_pred = predict(classifier, newdata = test_set[, -3])
#y_pred = ifelse(y_pred > 0.8, 1, 0)
y_pred

# Making the Confusion Matrix
table(unlist(test_set[, 3]), y_pred)

summary(classifier)
# Plotting the tree
plot(classifier)
text(classifier)

# Visualising the Training set results : Logistic regression
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by = 0.5)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by = 0.5)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Customer ID.x', 'Input Probability')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Training set)',
     xlab = 'Customer ID.x', ylab = 'Input Probability', 
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

################################################################################

################################################################################
# Naive Bayes: Fitting Naive Bayes to the Training set
# P(A|B) = P(B|A) * P(A) / P(B)
################################################################################
library(e1071)
# Encoding the target feature as factor
training_set$`Output Probability` = factor(training_set$`Output Probability`, levels = c(0,1))
test_set$`Output Probability` = factor(test_set$`Output Probability`, levels = c(0,1))

classifier = naiveBayes(x = training_set[, -3],
                        y = training_set$`Output Probability`)

# predicting the Test set results
y_pred = predict(classifier, newdata = test_set[, -3])
y_pred

# Making the Confusion Matrix
table(unlist(test_set[, 3]), y_pred)

summary(classifier)
# Plotting the tree
plot(classifier)
text(classifier)

# Visualising the Training set results : Logistic regression
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by = 0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Customer ID.x', 'Input Probability')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Naive Bayes (Training set)',
     xlab = 'Customer ID.x', ylab = 'Input Probability', 
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

################################################################################

################################################################################
# Decision Tree Classification
################################################################################
library(rpart)
# Encoding the target feature as factor
training_set$`Output Probability` = factor(training_set$`Output Probability`, levels = c(0,1))
test_set$`Output Probability` = factor(test_set$`Output Probability`, levels = c(0,1))

classifier = rpart(formula = `Output Probability` ~ .,
                   data = training_set)

# predicting the Test set results
y_pred = predict(classifier, newdata = test_set[, -3], type = 'class')
y_pred

# Making the Confusion Matrix
table(unlist(test_set[, 3]), y_pred)

# plotting the decision tree
summary(classifier)
# Plotting the tree
plot(classifier)
text(classifier)

# Visualising the Training set results : Decision Tree Classification
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by = 0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Customer ID.x', 'Input Probability')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3], main = 'Decision Tree (Training set)',
     xlab = 'Customer ID.x', ylab = 'Input Probability', 
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

################################################################################

################################################################################
# Random Forest Classification
################################################################################
library(randomForest)
# Encoding the target feature as factor
training_set$`Output Probability` = factor(training_set$`Output Probability`, levels = c(0,1))
test_set$`Output Probability` = factor(test_set$`Output Probability`, levels = c(0,1))

classifier = randomForest(x = training_set[, -3],
                          y = training_set$`Output Probability`,
                          ntree = 500)

# predicting the Test set results
y_pred = predict(classifier, newdata = test_set[, -3], type = 'class')
y_pred

# Making the Confusion Matrix
table(unlist(test_set[, 3]), y_pred)

# plotting the decision tree
summary(classifier)
# Plotting the tree
plot(classifier)
text(classifier)

# Visualising the Training set results : Decision Tree Classification
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) -1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) -1, max(set[, 2]) + 1, by = 0.1)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Customer ID.x', 'Input Probability')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Random Forest Classification (Training set)',
     xlab = 'Customer ID.x', ylab = 'Input Probability', 
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid ==1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

################################################################################

# **************************************************************************** #
# CONFUSION MATRIX
# **************************************************************************** #
cm = table(test_set[, 3], y_pred > 0.5)
################################################################################


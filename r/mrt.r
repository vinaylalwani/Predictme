

# install.packages("mlr", dependencies = TRUE)
# install.packages("dplyr", dependencies=TRUE)
# install.packages("data.table", dependencies=TRUE)
# install.packages("sqldf", dependencies = TRUE)
require(data.table)
orig_w1_data=fread('/Users/vinaylalwani/per/projects/ml/SR/W1.csv')
orig_w12_data=fread('/Users/vinaylalwani/per/projects/ml/SR/W12.csv')
w1_data = orig_w1_data[`Sales ID` != '#N/A' & `State` != 'Unspecified'][order(`Sales ID`, `Customer ID`)][, c("Creator ID"):=NULL]
w1_data = w1_data[, ("Sales ID") := lapply(.SD, as.numeric), .SDcols = "Sales ID"]




# Encoding categorical data
w1_data$Country = factor(w1_data$Country, levels = c("United States", "Canada"), labels = c(1, 0))
w1_data$State = factor(w1_data$State, 
                       levels = c("CA","NJ","OH","IL","AR","MD","FL","MA","GA","WA","NY","MI","IN","AZ","CO","TX","NB","ON","VA","NC","QC","PA","AL","BC","SC","KS","DC","OK","OR","LA","CT","AB","MO","WV","NE","TN","RI","NM","NV","MN","SK","ME","WI","VT","NH","KY","MB","IA","ID","DE","ND","HI","UT","MT"), 
                       labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54))

w12_data = orig_w12_data[`Sales ID` != '#N/A' & `State` != 'Unspecified'][order(`Sales ID`, `Customer ID`)][, c("Opportunity Created by"):=NULL]
w12_data$Country = factor(w12_data$Country, levels = c("United States", "Canada"), labels = c(1, 0))
w12_data$State = factor(w12_data$State, 
                       levels = c("CA","NJ","OH","IL","AR","MD","FL","MA","GA","WA","NY","MI","IN","AZ","CO","TX","NB","ON","VA","NC","QC","PA","AL","BC","SC","KS","DC","OK","OR","LA","CT","AB","MO","WV","NE","TN","RI","NM","NV","MN","SK","ME","WI","VT","NH","KY","MB","IA","ID","DE","ND","HI","UT","MT"), 
                       labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54))

# intersection of w1 and w12 only
successData = merge(w1_data, w12_data)[order(`Sales ID`, `Customer ID`)]
#mergeData = merge(w1_data, w12_data, by = c("Opportunity ID"), all = TRUE)
mergeData = merge(w1_data, w12_data, 
                  by = c("Opportunity ID"), all = TRUE)[, c("Country.x", "State.x", "Customer ID.y", "Sales ID.y", "Country.y", "State.y"):=NULL]

# get the number transactions of a given sales id
data.frame(table(mergeData$`Sales ID.x`))

# convert all NA to zeroes to prepare the training set
mergeData[is.na(mergeData)] <- 0
# choose a sales id to run the algorithm
sampleDataSet = mergeData[`Sales ID.x`==10034]

# prepare the training and test dataset
library(caTools)
set.seed(123)
split = sample.split(sampleDataSet$`Output Probability`, SplitRatio = 0.8)
training_set = subset(sampleDataSet, split == TRUE)
test_set = subset(sampleDataSet, split == FALSE)

# Feature scaling
#training_set = scale(training_set)
#test_set = scale(test_set)

# fit linear regression to the training set
regressor = lm(formula = `Output Probability` ~ `Input Probability` + `Customer ID.x` + `Predicted Revenue`,
               data = training_set)
regressor = lm(formula = `Output Probability` ~ `Input Probability`,
               data = training_set)
summary(regressor)
# Plotting the tree
plot(regressor)
text(regressor)


# predict
y_pred = predict(regressor, newdata = test_set)
y_pred


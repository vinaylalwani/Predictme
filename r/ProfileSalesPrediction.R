# **************************************************************************** #
#                 READ THE DATA
# **************************************************************************** #
require(data.table)
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

uniqSalesId = data.frame(table(mergeData$`Sales ID.x`))[1]
#firstFive = head(uniqSalesId,n=5)
#firstFiveList = c(firstFive["Var1"])
uniqSalesIdList = c(uniqSalesId["Var1"])

# iterate thru all the salesid and perform classification
for (sId in uniqSalesIdList$Var1){
  # convert all NA to zeroes to prepare the training set
  mergeData[is.na(mergeData)] <- 0
  # choose a sales id to run the algorithm; working: 10053, 10062; not working: 10032; tbd: 10043
  sampleDataSet = mergeData[`Sales ID.x`==paste(sId)]
  #sampleDataSet = mergeData[`Sales ID.x`== 10024]
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
  # make an 80-20 split
  #split = sample.split(sampleDataSet$`Output Probability`, SplitRatio = 0.8)
  #training_set = subset(sampleDataSet, split == TRUE)
  #test_set = subset(sampleDataSet, split == FALSE)
  
  # use a hard coded test dataset which varies from 10 % to 50 % input probability
  training_set = sampleDataSet
  test_set = fread('/Users/vinaylalwani/per/projects/ml/SR/testset.csv')
  

  # **************************************************************************** #
  
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
  print(sId)
  print(y_pred)
}

write.table(y_pred, file = "/Users/vinaylalwani/per/projects/ml/SR/result.csv", sep = ",", qmethod = "double", row.names=FALSE)
final_df <- as.data.frame(t(fread('/Users/vinaylalwani/per/projects/ml/SR/result.csv')))
final_df
# Making the Confusion Matrix
table(unlist(test_set[, 3]), y_pred)


################################################################################

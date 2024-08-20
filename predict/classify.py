import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor

# **************************************************************************** #
#                 READ THE DATA
# **************************************************************************** #

orig_w1_data = pd.read_csv('/Users/vinaylalwani/per/projects/ml/SR/v9.0_W1.csv')
orig_w12_data = pd.read_csv('/Users/vinaylalwani/per/projects/ml/SR/v9.0_W12.csv')

w1_data = pd.DataFrame(orig_w1_data, columns = ['Opportunity ID', 'Sales ID', 'Input Probability', ' Predicted Revenue'])
w1_data = w1_data[w1_data['Sales ID'].notna() & w1_data['Sales ID'].notnull() &
                  w1_data['Opportunity ID'].notna() & w1_data['Opportunity ID'].notnull()].sort_values('Sales ID')
print("w1_data:", len(w1_data))

w12_data = pd.DataFrame(orig_w12_data, columns = ['Opportunity ID', 'Sales ID', 'Output Probability', ' Final Revenue'])
w12_data = w12_data[w12_data['Sales ID'].notna() & w12_data['Sales ID'].notnull() &
                    w12_data['Opportunity ID'].notna() & w12_data['Opportunity ID'].notnull()].sort_values('Sales ID')
print("w12_data:", len(w12_data))

# **************************************************************************** #
#                 PRE PROCESS THE DATA
# **************************************************************************** #

# intersection of w1 and w12 only, order by sales id
# successData = merge(w1_data, w12_data)[order(`Sales ID`, `Customer ID`)]
# successData = pd.merge(w1_data, w12_data)
# print("successData:", len(successData))
# print(successData)

# merge the input and output records
mergeData = pd.merge(w1_data, w12_data, on = "Opportunity ID", how='right')
print("mergeData:", len(mergeData))
# print(mergeData)

# get list of sales ids
uniqSalesIdList = mergeData['Sales ID_x'].unique()
# print("uniqSalesIdList:", uniqSalesIdList)
# uniqSalesIdListSize = mergeData.groupby(['Sales ID_x']).size()
# print("uniqSalesIdListSize:", uniqSalesIdListSize)

# iterate thru all the salesid and perform classification
for sId in uniqSalesIdList:
    # sId = 10023
    mergedDataSet = mergeData[mergeData['Sales ID_x'] == sId]
    # remove rows whose input probability >= 50%
    mergedDataSet = mergedDataSet[mergedDataSet['Input Probability'] >= 50]
    # print(mergedDataSet)
    if len(mergedDataSet) > 5:
        # print("processing ", sId, "; length=", len(mergedDataSet))

        #  probability in decimal format
        mergedDataSet['Input Probability'] = mergedDataSet['Input Probability'] / 100
        mergedDataSet['Output Probability'] = mergedDataSet['Output Probability'] / 100

        # consider required params only
        # mergedDataSet = pd.DataFrame(mergedDataSet, columns =
        #     ['Opportunity ID', 'Sales ID_x', 'Input Probability', 'Output Probability', ' Predicted Revenue', ' Final Revenue'])

        # create sampleDataSet
        sampleDataSet = pd.DataFrame(mergedDataSet, columns =
            ['Input Probability', ' Predicted Revenue', 'Output Probability'])

        # **************************************************************************** #
        #     TRAINING AND TEST SET
        # **************************************************************************** #
        # prepare the training and test dataset
        # make an 80-20 split
        training_set, test_set = train_test_split(sampleDataSet, test_size=0.2)
        print("training_set (for comparision) for ", sId, " ", training_set['Output Probability'].tolist())
        print("test_set (for comparision) for ", sId, " ", test_set['Output Probability'].tolist())

        # Create a Gaussian Classifier
        model = GaussianNB()
        # Train the model using the training sets
        model.fit(training_set[['Input Probability', ' Predicted Revenue']], training_set[['Output Probability']].values.ravel())
        # Predict Output
        predicted = model.predict(test_set[['Input Probability', ' Predicted Revenue']])
        print("Prediction (Gaussian naive Bayes) for :", sId, " ", predicted)

        # Create a Decision Tree Regressor
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(training_set[['Input Probability', ' Predicted Revenue']], training_set[['Output Probability']].values.ravel())
        # Predict Output
        predicted = regressor.predict(test_set[['Input Probability', ' Predicted Revenue']])
        print("Prediction (DecisionTree) for :", sId, " ", predicted)


        # print(test_set['Output Probability'].tolist())
        # print(test_set[['Output Probability']].values.tolist())
        # print(pd.Series(test_set[['Output Probability']]), name='Actual')
        # pd.Series((test_set[['Output Probability']]), name='Actual'), pd.Series(predicted, name='Predicted')))
        # print("Confusion Matrix: ", pd.crosstab(pd.Series((test_set['Output Probability'].values.tolist()), name='Actual'), pd.Series(predicted, name='Predicted')))


        print("------------------------------------------------------")






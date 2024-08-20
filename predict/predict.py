import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="sklearn", message="^internal gelsd")

class Config(object):
    TEST_SIZE = 0.2 # 20%
    CV = 2          # n fold cross validation
    DO_CV = False
    B1 = "0% - 30%"
    B2 = "31% - 50%"
    B3 = "51% - 70%"
    B4 = "71% - 100%"
    B_ALL = "0% - 100%"

class Predict:
    def __init__(self):
        pass

    # **************************************************************************** #
    #     XGBoost
    # **************************************************************************** #
    def run_XG_Boost(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        # Fitting XGBoost to the Training set
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        print('--------------------\n0. XGBoost (C): y_pred', y_pred)
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        classifier.fit(X, y)
        y_pred = classifier.predict(myX_test)
        print('   XGBoost for ', myX_test, ' : prediction= ', y_pred)

    # **************************************************************************** #
    #     Kernel SVM (apple/orange analogy); Hyperplane
    # **************************************************************************** #
    def run_SVM(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        # Fitting Kernel SVM to the training set
        # non linear (squared exponential, rbf(radial basis)/poly/sigmoid/precomputed) / linear
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print('--------------------\n1. Kernel SVM (C): y_pred', y_pred)
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        classifier.fit(X, y)
        y_pred = classifier.predict(myX_test)
        print('   Kernel SVM for ', myX_test, ' : prediction= ', y_pred)

        # # Applying Grid search to find the best model and the best parameters
        # # C is the penalty parameter of the error term; the more it is would prevent overfitting,
        # # the greater the value might landup to underfitting.
        # parameters = [
        #     {'C':[1,10,100,1000], 'kernel' : ['linear']},
        #     {'C':[1,10,100,1000], 'kernel' : ['sigmoid'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}
        # ]
        # grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv = Config.CV, n_jobs= -1)
        # grid_search = grid_search.fit(X_train, y_train)
        # best_accuracy = grid_search.best_score_   # is the mean from k-fold
        # best_parameters = grid_search.best_params_
        # print('best_accuracy=', best_accuracy, '; best_parameters=', best_parameters)

    # **************************************************************************** #
    #     Logistic Regression
    # **************************************************************************** #
    def run_log_reg(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        # Fitting simple Linear Regression to the training set
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print('--------------------\n2. Logistic Regression: y_pred', y_pred)
        # TODO: saw issues sometimes due to handling a mix of binary and continuous targets
        # print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=regressor, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        regressor.fit(X, y)
        y_pred = regressor.predict(myX_test)
        print('   Logistic Regression for ', myX_test, ' : prediction= ', y_pred)

    # **************************************************************************** #
    #     Decision Tree Regression
    # **************************************************************************** #
    def run_decision_tree_regressor(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print('--------------------\n3. Decision Tree (R):', y_pred)
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=regressor, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        regressor.fit(X, y)
        y_pred = regressor.predict(myX_test)
        print('   Decision Tree (R) for ', myX_test, ' : prediction= ', y_pred)

    # **************************************************************************** #
    #     Decision Tree Classifier
    # **************************************************************************** #
    def run_decision_tree_classifier(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print('--------------------\n4. Decision Tree (C):', y_pred)
        print('   Confusion Matrix: ')
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        classifier.fit(X, y)
        y_pred = classifier.predict(myX_test)
        print('   Decision Tree (C) for ', myX_test, ' : prediction= ', y_pred)

    # **************************************************************************** #
    #     Random Forest Regression
    # **************************************************************************** #
    def run_random_forest_regressor(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        # n_estimators: no. of trees in the forest; criterion: function to measure the quality of a split, default - mse, mean squared error
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        print('--------------------\n5. Random Forest (R): y_pred', y_pred)
        print('   Confusion Matrix: ')
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=regressor, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        regressor.fit(X, y)
        y_pred = regressor.predict(myX_test)
        print('   Random Forest (R) for ', myX_test, ' : prediction= ', y_pred)


    # **************************************************************************** #
    #     Random Forest Classification
    # **************************************************************************** #
    def run_random_forest_classifier(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        # n_estimators: no. of trees in the forest; criterion: function to measure the quality of a split, default - mse, mean squared error
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print('--------------------\n6. Random Forest (C): y_pred', y_pred)
        print('   Confusion Matrix: ')
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        classifier.fit(X, y)
        y_pred = classifier.predict(myX_test)
        print('   Random Forest (C) for ', myX_test, ' : prediction= ', y_pred)


    # **************************************************************************** #
    #     Naive Bayes Classification
    # **************************************************************************** #
    def run_naive_bayes_classifier(self, X, y, X_train, y_train, X_test, y_test, myX_test):
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print('--------------------\n7. Naive Bayes (C): y_pred', y_pred)
        print('   Confusion Matrix: ')
        print(confusion_matrix(y_test, y_pred))

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=Config.CV)
        print('̄x = ', accuracies.mean(), 'σ = ', accuracies.std())

        classifier.fit(X, y)
        y_pred = classifier.predict(myX_test)
        print('   Naive Bayes (C) for ', myX_test, ' : prediction= ', y_pred)


    #################
    # main function
    #################

    def main(self):
        # **************************************************************************** #
        #                 READ THE DATA
        # **************************************************************************** #

        # pd.options.display.float_format = '{:,.0f}'.format
        orig_w1_data = pd.read_csv('files/v9.0_W1.csv')
        orig_w12_data = pd.read_csv('files/v9.0_W12.csv')

        w1_data = pd.DataFrame(orig_w1_data, columns = ['State', 'Opportunity ID', 'Sales ID', 'Opportunity Created by', 'Input Probability', ' Predicted Revenue'])
        w1_data = w1_data.drop(columns=['State', 'Opportunity Created by'])
        w1_data = w1_data[w1_data['Sales ID'].notna() & w1_data['Sales ID'].notnull() &
                          w1_data['Opportunity ID'].notna() & w1_data['Opportunity ID'].notnull()].fillna(0.0).astype(int)
        print("w1_data:", len(w1_data))
        w1_data.columns = ['OppID', 'SalesID', 'Prob', 'Rev']

        w12_data = pd.DataFrame(orig_w12_data, columns = ['State', 'Opportunity ID', 'Sales ID', 'Opportunity Created by', 'Output Probability', ' Final Revenue'])
        w12_data = w12_data.drop(columns=['State', 'Opportunity Created by'])
        w12_data = w12_data[w12_data['Sales ID'].notna() & w12_data['Sales ID'].notnull() &
                            w12_data['Opportunity ID'].notna() & w12_data['Opportunity ID'].notnull()].fillna(0.0).astype(int)
        print("w12_data:", len(w12_data))
        # print(pd.DataFrame(w12_data[w12_data['Sales ID'] == 10023]))
        w12_data.columns = ['OppID', 'SalesID', 'Prob', 'Rev']

        # **************************************************************************** #
        #                 PRE PROCESS THE DATA
        # **************************************************************************** #

        # merge the input and output records
        merge_data = pd.merge(w1_data, w12_data, on = ["OppID", "Rev"], how='left')
        print("mergeData:", len(merge_data))
        merge_data = merge_data.fillna(0.0).astype(int)
        merge_data.drop(['SalesID_y'], axis = 1, inplace = True)
        merge_data.columns = ['OppID', 'SID', 'IP', 'Rev', 'OP']
        # print(mergeData)

        # get and list of sales ids by size
        uniqSalesIdList = merge_data['SID'].unique()
        uniqSalesIdListSize = merge_data.groupby(['SID']).size().sort_values(ascending=False)
        # print("SalesId Txn count: \n", uniqSalesIdListSize)

        # iterate thru all the salesid and perform classification
        # uniqSalesIdList = [10058]  # test only: comment this to iterate thru all the salesid
        uniqSalesIdList = [10001]  # test only: comment this to iterate thru all the salesid
        # uniqSalesIdList = [10063]  # test only: comment this to iterate thru all the salesid
        # uniqSalesIdList = [10032]  # test only: comment this to iterate thru all the salesid

        ###################################################################################################
        # loop thru salesid and create models for each algorithm to identify the accuracy and variation
        ###################################################################################################

        for sId in uniqSalesIdList:
            merged_data_set = merge_data[merge_data['SID'] == sId]
            print("--------------------\nProcessing ", sId, "; Length=", len(merged_data_set))

            #################################################
            # create buckets based on input probabilities
            #################################################
            b1 = merged_data_set.loc[merged_data_set['IP'].isin(range(0, 31))]  # 0% - 30%
            b2 = merged_data_set.loc[merged_data_set['IP'].isin(range(31, 51))]  # 31% - 50%
            b3 = merged_data_set.loc[merged_data_set['IP'].isin(range(51, 71))]  # 51% - 70%
            b4 = merged_data_set.loc[merged_data_set['IP'].isin(range(71, 101))]  # 71% - 100%
            print("No. of opportunities based on input probability")
            print(Config.B1, len(b1))
            print(Config.B2, len(b2))
            print(Config.B3, len(b3))
            print(Config.B4, len(b4))

            # for bucket'ized dataset
            # merged_data_set = b1
            # run all data
            if len(merged_data_set) > 5:
                print("--------------------\nProcessing bucket", Config.B_ALL, " for sId:", sId, "; No. or records=", len(merged_data_set))

                # create sampleDataSet
                data_set = pd.DataFrame(merged_data_set, columns=['IP', 'Rev', 'OP'])
                # print('sampleDataSet', sampleDataSet)

                X = data_set.iloc[:, :-1].values  # remove all columns except for the dependant variable
                y = data_set.iloc[:, 2].values  # choose the dependant variable index

                # **************************************************************************** #
                #     TRAINING AND TEST SET
                # **************************************************************************** #
                # spot check
                myX_test = [[60., 973.]]

                # prepare the training and test dataset
                # make an 80-20 split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=0,
                                                                    shuffle=True)
                # print('X\n', X)
                # print('y\n', y)
                print('X_train\n', X_train)
                print('y_train', y_train, '; size=', len(y_train))
                print('X_test\n', X_test)
                print('y_test', y_test, '; size=', len(y_test))

                if len(set(y_train)) > 1 and len(set(y_test)) > 1:
                    ## Feature scaling
                    # x = StandardScaler().fit(X_train)
                    # print("*********** feature scaler start ***********")
                    # print(X_train)
                    # print(x)
                    # print(x.scale_)
                    # x1 = print(x.transform(X_train))
                    # print(x1)
                    # print("*********** feature scaler end ***********")

                    self.run_XG_Boost(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_SVM(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_log_reg(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_decision_tree_regressor(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_decision_tree_classifier(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_random_forest_regressor(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_random_forest_classifier(X, y, X_train, y_train, X_test, y_test, myX_test)
                    self.run_naive_bayes_classifier(X, y, X_train, y_train, X_test, y_test, myX_test)

                else:
                    print('********** ignoring ', sId,
                          ' due to insufficient classes (combination of probabilities)')
                    print('set(y_train) = ', set(y_train))
                    print('set(y_test) = ', set(y_test))
            else:
                print('********** ignoring ', sId, ' due to insufficient data ', len(merged_data_set))

                # **************************************************************************** #
                #     K-Means
                # **************************************************************************** #
                # determine the number of clusters
                # import matplotlib.pyplot as plt
                # wcss = []
                # for i in range(1, 11):
                #     kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
                #     kmeans.fit(X)
                #     wcss.append(kmeans.inertia_)
                # plt.plot(range(1, 11), wcss)
                # plt.title('The Elbow Method')
                # plt.xlabel('Number of clusters')
                # plt.ylabel('WCSS')
                # plt.show()

                # kmeans = KMeans(n_clusters=2, init = 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
                # y_kmeans = kmeans.fit_predict(X)
                # plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c= 'red', label = 'Cluster 1')
                # plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c= 'blue', label = 'Cluster 2')
                # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s= 300, c = 'yellow', label = 'Centroids')
                # plt.title('Cluster of clients')
                # plt.xlabel('X')
                # plt.ylabel('Y')
                # plt.legend()
                # plt.show()

                # plt.scatter(X_train, y_train, color='red')
                # plt.plot(X_train, regressor.predict(X_train), color='blue')
                # plt.title('linear regression')
                # plt.xlabel('independent variables')
                # plt.ylabel('dependant variables')
                # plt.show()

                # # Create a Gaussian Classifier
                # model = GaussianNB()
                # # Train the model using the training sets
                # model.fit(training_set[['Input Probability', ' Predicted Revenue']], training_set[['Output Probability']].values.ravel())
                # # Predict Output
                # predicted = model.predict(test_set[['Input Probability', ' Predicted Revenue']])
                # print("Prediction (Gaussian naive Bayes) for :", sId, " ", predicted)
                #
                # # Create a Decision Tree Regressor
                # regressor = DecisionTreeRegressor(random_state=0)
                # regressor.fit(training_set[['Input Probability', ' Predicted Revenue']], training_set[['Output Probability']].values.ravel())
                # # Predict Output
                # predicted = regressor.predict(test_set[['Input Probability', ' Predicted Revenue']])
                # print("Prediction (DecisionTree) for :", sId, " ", predicted)


                # print(test_set['Output Probability'].tolist())
                # print(test_set[['Output Probability']].values.tolist())
                # print(pd.Series(test_set[['Output Probability']]), name='Actual')
                # pd.Series((test_set[['Output Probability']]), name='Actual'), pd.Series(predicted, name='Predicted')))
                # print("Confusion Matrix: ", pd.crosstab(pd.Series((test_set['Output Probability'].values.tolist()), name='Actual'), pd.Series(predicted, name='Predicted')))


                # print("------------------------------------------------------")


if __name__ == '__main__':
    predict = Predict()
    predict.main()


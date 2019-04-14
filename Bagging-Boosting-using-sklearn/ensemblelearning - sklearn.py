# -*- coding: utf-8 -*-

"""
Implementation of bagging and boosting methods using scikit-learn

@author: Vamshider Reddy Voncha
    
"""
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Bagging
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification

def sk_bagging(dataset_name, depth=5, num_trees=10):
    print("sk_bagging", end='\n')
     # Load the train data
    data_train = get_data(dataset_name, '.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Load the test data
    data_test = get_data(dataset_name, '.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
    clf = RandomForestClassifier(n_estimators=num_trees, max_depth=depth,
                                 criterion='entropy', random_state=0)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test);
     
    #err_train = 1 - clf.score(X_train, y_train)
    #err_test = 1 - clf.score(X_test, y_test)    
    # error
    #print_errors(err_train, err_test, dataset_name, "Errors for bagging using sklearn on")        
    # confusion matrix
    print_confusion_matrix(y_test, y_test_pred, dataset_name, "Confusion matrix for bagging using sklearn on")
    print(end='\n')


#------------------------------------------------------------------------------
# Boosting
#------------------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets import make_classification
def sk_boosting(dataset_name, depth=1, num_stumps=10):
    print("sk_boosting", end='\n')
    # Load the train data
    data_train = get_data(dataset_name, '.train')
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Load the test data
    data_test = get_data(dataset_name, '.test')
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
    base = DecisionTreeClassifier(max_depth=depth)
    clf = AdaBoostClassifier(base_estimator = base, n_estimators=num_stumps)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
     
    #err_train = 1 - clf.score(X_train, y_train)
    #err_test = 1 - clf.score(X_test, y_test)    
    # error
    #print_errors(err_train, err_test, dataset_name, "Errors for boosting using sklearn on")        
    # confusion matrix
    print_confusion_matrix(y_test, y_test_pred, dataset_name, "Confusion matrix for boosting using sklearn on")
    print(end='\n')
 
    
#------------------------------------------------------------------------------
#Computes the average error between the true labels (y_true) and the predicted 
#   labels (y_pred)
#Returns the error = (1/n) * sum(y_true != y_pred)
#------------------------------------------------------------------------------
def compute_error(y_true, y_pred):
    return (1/y_true.size)*(np.sum(y_true != y_pred))

#------------------------------------------------------------------------------
# Compute confusion matrix
#------------------------------------------------------------------------------
def confusion_matrix(true_label, predicted_label):
    labels = np.unique(true_label)
    matrix = [[0 for x in range(len(labels))] for y in range(len(labels))]
    for t, p in zip(true_label, predicted_label):
        #matrix[t][p] += 1
        if t == 1 and p == 1:
            matrix[0][0] += 1
        elif t == 0 and p == 0:
            matrix[1][1] += 1
        elif t == 1 and p == 0:
            matrix[0][1] += 1
        elif t == 0 and p == 1:
            matrix[1][0] +=1
            
    return matrix    # Load the training data

#------------------------------------------------------------------------------
# print confusion matrix
#------------------------------------------------------------------------------
def print_confusion_matrix(true_label, predicted_label, 
                           dataset_name = "dummy", context = "classifier"):
    print("\n "+ context + " " + dataset_name + " dataset")
    df = pd.DataFrame(
        confusion_matrix(true_label, predicted_label),
        columns=['Predicted Positive', 'Predicted Negative'],
        index=['Actual Positive', 'Actual Negative']
    )
    print(df)

#------------------------------------------------------------------------------
# print confusion matrix
#------------------------------------------------------------------------------
def print_errors(on_train_data, on_test_data, 
                 dataset_name = "dummy", context = "classifier"):
    print("\n "+ context + " " + dataset_name + " ERRORS")        
    print('Train Error = {0:4.2f}%.'.format(on_train_data * 100))
    print('Test Error = {0:4.2f}%.'.format(on_test_data * 100))
    
#------------------------------------------------------------------------------
# getdata
#------------------------------------------------------------------------------
def get_data(dataset_name='dummy', extension='.data'):
    data_path = './'
    return np.genfromtxt((data_path + dataset_name + extension), 
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)

         
#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load the training data
    dataset_name = 'mushroom'  
 
    # bagging with scikit-learn
    sk_bagging(dataset_name, 3, 10)
    
    sk_bagging(dataset_name, 5, 10)
    
    sk_bagging(dataset_name, 3, 20)
    
    sk_bagging(dataset_name, 5, 20)
  
    #boosting with scikit-learn
    sk_boosting(dataset_name, 1, 20)
    
    sk_boosting(dataset_name, 2, 20)
    
    sk_boosting(dataset_name, 1, 40)
    
    sk_boosting(dataset_name, 2, 40)
    
#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# # Data Set Up
# Welcome to my final project assignment for COGS118A. 
# 

# In[58]:


bank_df = pd.read_csv("bank+marketing/bank/bank-full.csv", sep = ";")
display(bank_df)


# In[59]:


mushroom_df = pd.read_csv("mushroom/agaricus-lepiota.data")
display(mushroom_df)


# In[60]:


income_df = pd.read_csv("census+income/adult.data", delimiter=",")
display(income_df)


# ###  One-Hot Encoding 
# 
# While our data looks nice and formatted, we sadly cannot work with categorical labels to perform our model training. To resolve this, we utilize one-hot encoding to transform the set of each possible nominal values from every column into a new column span that treats each label as a 0 or 1. For other columns with numerical values, we simply treat their values the same. Thankfully, numpy comes with a method to do this for us automatically for each of our datasets

# In[61]:


bank_encoded = pd.get_dummies(bank_df)
bank_encoded = bank_encoded.replace({True: 1, False: 0})

mushroom_encoded = pd.get_dummies(mushroom_df)
mushroom_encoded = mushroom_encoded.replace({True: 1, False: 0})

income_encoded = pd.get_dummies(income_df)
income_encoded = income_encoded.replace({True: 1, False: 0})


# In[62]:


display(bank_encoded)


# In[63]:


display(mushroom_encoded)


# In[64]:


display(income_encoded)


# ### Converting to Matrices
# Before we rush ahead, we need to convert each dataframe into a form that we can easily work with so we can utilize complex mathematical operations. We'll extract each value in the dataframes and transfer them over to a numpy array

# In[65]:


bank = bank_encoded.values
mushroom = mushroom_encoded.values
income = income_encoded.values

# Verifying that the shape matches their dataframe shape
print(bank.shape)
print(mushroom.shape)
print(income.shape)


# ## Cleaning Up the Data
# With the data mostly done, we'll focus on setting up the classifiers. We also have to address the fact that we can't keep the columns that classify our observations since they are not features. Since we already have each table in matrix form, we can easily remove and extract our labels for each data set! Additionally, we must shuffle the data.

# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[67]:


# Extract the corresponding column for the labels, then remove the last two columns 
bank_labels = (bank[:,-1]).reshape(-1,1).astype(float)
bank = bank[:,: -2]


mushroom_labels = (mushroom[:,1]).reshape(-1,1).astype(float)
mushroom = mushroom[:, 2:]

income_labels = (income[:,-1]).reshape(-1,1).astype(float)
income = income[:,:-2]

print('Bank Shape: {}'.format(bank.shape))
print('Mushroom Shape: {}'.format(mushroom.shape))
print('Income Shape: {}'.format(income.shape))

# print(bank_labels.shape)
# print(mushroom_labels.shape)
# print(income_labels.shape)


# Convert every false (0) to -1 in our labels arrays
unique, counts = np.unique(mushroom_labels, return_counts=True)
count_dict = dict(zip(unique, counts))
print(count_dict) # {0: 7, 1: 4, 2: 1, 3: 2, 4: 1}

bank_labels[bank_labels == 0] = -1
mushroom_labels[mushroom_labels == 0] = -1
income_labels[income_labels == 0] = -1


# print(bank_labels) 
# print(mushroom_labels) 
# print(income_labels)


# Stack the labels with their original tables to shuffle 
bank = np.hstack((bank, bank_labels)) 
mushroom = np.hstack((mushroom, mushroom_labels)) 
income = np.hstack((income, income_labels)) 

np.random.seed(1)          
np.random.shuffle(bank) 
np.random.shuffle(mushroom) 
np.random.shuffle(income) 



# In[68]:


# For class weights 
bank_unique, bank_counts = np.unique(bank_labels, return_counts=True)
bank_count_dict = dict(zip(bank_unique, bank_counts))

income_unique, income_counts = np.unique(income_labels, return_counts=True)
income_count_dict = dict(zip(income_unique, income_counts))

mushroom_unique, mushroom_counts = np.unique(mushroom_labels, return_counts=True)
mushroom_count_dict = dict(zip(mushroom_unique, mushroom_counts))



# Extract features and labels for calculating proportions of each class
bank_x = bank[:,:-1]
bank_y = bank[:,-1]

income_x = income[:,:-1]
income_y = income[:,-1]

mushroom_x = mushroom[:,:-1]
mushroom_y = mushroom[:,-1]


# Create a dict of counts of each class for each dataset
bank_weight_1_neg = len(bank_x)/bank_count_dict.get(-1)
bank_weight_1_pos = len(bank_x)/bank_count_dict.get(1)
bank_class_weights = {-1: bank_weight_1_neg , 1: bank_weight_1_pos }  

income_weight_1_neg = len(income_x)/income_count_dict.get(-1)
income_weight_1_pos = len(income_x)/income_count_dict.get(1)
income_class_weights = {-1: income_weight_1_neg , 1: income_weight_1_pos } 


mushroom_weight_1_neg = len(mushroom_x)/mushroom_count_dict.get(-1)
mushroom_weight_1_pos = len(mushroom_x)/mushroom_count_dict.get(1)
mushroom_class_weights = {-1: mushroom_weight_1_neg , 1: mushroom_weight_1_pos } 


# # Support Vector Machines
# 
# 
# We won't use the default SVM library but rather SVCLinear. It is similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.The main differences between LinearSVC and SVC lie in the loss function used by default, and in the handling of intercept regularization between those two implementations.
# 
# The goal here is to find the best hyperparameter C 

# In[69]:


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[70]:


# Hyperparamter list
C_list = [0.1, 1, 10, 100, 1000, 10000]


# In[71]:


# Draw heatmaps for result of grid search.
def draw_heatmap(errors, param_list, title):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(errors, annot=True, fmt='.4f', yticklabels=param_list, xticklabels=[])
    ax.collections[0].colorbar.set_label('error')
    ax.set(ylabel='hyper parameter')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    plt.show()


# In[72]:


def calcSVCMetrics(X_train,X_test, Y_train,Y_test, C_List, class_weights):

    clf = LinearSVC(dual=False, class_weight=class_weights)
    param_grid = {'C': C_list}

    
    # Perform 3-Fold cross validation for each hyperparameter
    grid_search = GridSearchCV(clf, param_grid, cv=3, return_train_score=True )  
    
    # Fit the model
    grid_search.fit(X_train, Y_train)    
    
    # Gather the results
    opt_C = grid_search.best_params_['C']  
                                                 
                                        
    cross_validation_accuracies =  grid_search.cv_results_['mean_test_score']
    cross_validation_errors = 1 - cross_validation_accuracies.reshape(-1,1)
    
    mean_training_accuracies = grid_search.cv_results_['mean_train_score']
    mean_training_errors = 1 - mean_training_accuracies.reshape(-1,1)
    
    Y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(Y_pred, Y_test)
    test_error = 1 - sum(Y_pred == Y_test) / len(X_test)

    
    return opt_C,cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error


# ## SVM for BANK 

# In[79]:


# Array to track accuracies for each partition
svm_bank_accs = []

# 20% Training and 80% Testing 
bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 = train_test_split(bank_x, bank_y, test_size=0.8)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80, C_list, bank_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_bank_accs.append(test_accuracy)


# In[80]:


# 50% Training and 50% Testing 
bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50 = train_test_split(bank_x, bank_y, test_size=0.5)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50, C_list, bank_class_weights )

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_bank_accs.append(test_accuracy)


# In[81]:


# 80% Training and 20% Testing 
bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20 = train_test_split(bank_x, bank_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20, C_list, bank_class_weights  )

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_bank_accs.append(test_accuracy)


# ## SVM For INCOME
# 

# In[85]:


svm_income_accs = []
# 20% Training and 80% Testing 
income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80, C_list, income_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_income_accs.append(test_accuracy)


# In[86]:


# 50% Training and 50% Testing 
income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50, C_list, income_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_income_accs.append(test_accuracy)


# In[87]:


# 80% Training and 20% Testing 
income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20, C_list, income_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))

svm_income_accs.append(test_accuracy)


# ## SVM For MUSHROOM

# In[88]:


svm_mushroom_accs = []

# 20% Training and 80% Testing 
mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80, C_list, mushroom_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_mushroom_accs.append(test_accuracy)


# In[89]:


# 50% Training and 50% Testing 
mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50, C_list, mushroom_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_mushroom_accs.append(test_accuracy)


# In[90]:


# 80% Training and 20% Testing 
mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcSVCMetrics(mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20, C_list, mushroom_class_weights)

draw_heatmap(cross_validation_errors, C_list, title='cross-validation error w.r.t C')
print("Best C: {}".format(opt_C))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per C: {}".format(mean_training_errors))
print("Avg training accuracies per C: {}".format(mean_training_accuracies))
svm_mushroom_accs.append(test_accuracy)


# ## SVM Results
# 
# Now that we've performed training and testing on all three datasets and partitions, let's average all the test accuracies. We'll use this to help with our comparison against Caruana's findings. 
# 
# 

# In[91]:


average_accuracy = np.sum([a + b + c for a, b, c in zip(svm_bank_accs, svm_income_accs, svm_mushroom_accs)]) / 9
print(svm_bank_accs)
print(svm_income_accs)
print(svm_mushroom_accs)
print("Average SVM accuracy {}".format(average_accuracy))


# # K Nearest Neighbors

# In[92]:


import scipy
from matplotlib.colors import ListedColormap
from functools import partial
from sklearn.neighbors import KNeighborsClassifier

# Hyperparameter list of possible K's
# Becuase of the scope of the project, I capped it at 15 since KNN involves expensive operations
k_range = list(range(1, 16))


# In[93]:


def calcKNNMetrics(X_train,X_test, Y_train,Y_test,k_range):

    param_grid = dict(n_neighbors=k_range)
    clf = KNeighborsClassifier(algorithm = 'kd_tree', weights='distance')        

    grid_search = GridSearchCV(clf, param_grid, cv=3, return_train_score=True,verbose=1, )

    # Fit the model
    grid_search.fit(X_train, Y_train)    
    
    # Gather the results
    opt_K = grid_search.best_params_
                                                 
                                        
    cross_validation_accuracies =  grid_search.cv_results_['mean_test_score']
    cross_validation_errors = 1 - cross_validation_accuracies.reshape(-1,1)
    
    mean_training_accuracies = grid_search.cv_results_['mean_train_score']
    mean_training_errors = 1 - mean_training_accuracies.reshape(-1,1)
    
    Y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(Y_pred, Y_test)
    test_error = 1 - sum(Y_pred == Y_test) / len(X_test)

    
    return opt_K,cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error


# ## KNN for Bank

# In[94]:


knn_bank_accs = []
# 20% Training and 80% Testing 
bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 = train_test_split(bank_x, bank_y, test_size=0.8)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_bank_accs.append(test_accuracy)


# In[95]:


# 50% Training and 50% Testing 
bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50 = train_test_split(bank_x, bank_y, test_size=0.5)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_bank_accs.append(test_accuracy)


# In[96]:


bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20 = train_test_split(bank_x, bank_y, test_size=0.2)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20,k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_bank_accs.append(test_accuracy)


# ## KNN For INCOME

# In[97]:


knn_income_accs = []
# 20% Training and 80% Testing 
income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_income_accs.append(test_accuracy)


# In[98]:


income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_income_accs.append(test_accuracy)


# In[99]:


income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics( income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 , k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best C: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_income_accs.append(test_accuracy)


# ## KNN For MUSHROOM

# In[100]:


knn_mushroom_accs = []

# 20% Training and 80% Testing 
mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error   =  calcKNNMetrics(mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80,  k_range)


draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best K: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_mushroom_accs.append(test_accuracy)


# In[102]:


# 50% Training and 50% Testing 
mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_K, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50, k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best K: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_mushroom_accs.append(test_accuracy)


# In[103]:


# 80% Training and 20% Testing 
mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_C, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcKNNMetrics(mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20,k_range)

draw_heatmap(cross_validation_errors, k_range, title='cross-validation error w.r.t K')
print("Best K: {}".format(opt_K))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per K: {}".format(mean_training_errors))
print("Avg training accuracies per K: {}".format(mean_training_accuracies))

knn_mushroom_accs.append(test_accuracy)


# In[104]:


average_knn_accuracy = np.sum([a + b + c for a, b, c in zip(knn_bank_accs, knn_income_accs, knn_mushroom_accs)]) / 9
print(average_knn_accuracy)


# # Decision Tree

# In[106]:


import seaborn as sns
from sklearn import tree

D_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[107]:


def calcDTMetrics(X_train,X_test, Y_train,Y_test, D_list):

    estimator = tree.DecisionTreeClassifier(criterion='entropy',random_state = 1 )

    param_grid = {'max_depth': D_list}
    grid_search = GridSearchCV(estimator, param_grid, cv=3, return_train_score=True)

    # Fit the model
    grid_search.fit(X_train, Y_train)    
    
    # Gather the results
    opt_D = grid_search.best_params_['max_depth']
                                                 
                                        
    cross_validation_accuracies =  grid_search.cv_results_['mean_test_score']
    cross_validation_errors = 1 - cross_validation_accuracies.reshape(-1,1)
    
    mean_training_accuracies = grid_search.cv_results_['mean_train_score']
    mean_training_errors = 1 - mean_training_accuracies.reshape(-1,1)
    
    Y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(Y_pred, Y_test)
    test_error = 1 - sum(Y_pred == Y_test) / len(X_test)

    
    return opt_D ,cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error


# ## Decision Tree for BANK

# In[128]:


dt_bank_accs = []
# 20% Training and 80% Testing 
bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(bank_x_train_20, bank_x_test_80, bank_y_train_20, bank_y_test_80 , D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_bank_accs.append(test_accuracy)


# In[129]:


# 50% Training and 50% Testing 
bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(bank_x_train_50, bank_x_test_50, bank_y_train_50, bank_y_test_50, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_bank_accs.append(test_accuracy)


# In[130]:


# 80% Training and 20% Testing 
bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(bank_x_train_80, bank_x_test_20, bank_y_train_80, bank_y_test_20, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_bank_accs.append(test_accuracy)


# ## Decision Tree for INCOME

# In[114]:


dt_income_accs = []
# 20% Training and 80% Testing 
income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(income_x_train_20, income_x_test_80, income_y_train_20, income_y_test_80, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_income_accs.append(test_accuracy)


# In[115]:


# 50% Training and 50% Testing 
income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(income_x_train_50, income_x_test_50, income_y_train_50, income_y_test_50, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_income_accs.append(test_accuracy)


# In[116]:


# 80% Training and 20% Testing 
income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(income_x_train_80, income_x_test_20, income_y_train_80, income_y_test_20 , D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_income_accs.append(test_accuracy)


# ## Decision Tree for Mushroom

# In[117]:


dt_mushroom_accs = []

# 20% Training and 80% Testing 
mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80 = train_test_split(income_x, income_y, test_size=0.8)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(mushroom_x_train_20, mushroom_x_test_80, mushroom_y_train_20, mushroom_y_test_80, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_mushroom_accs.append(test_accuracy)


# In[118]:


# 50% Training and 50% Testing 
mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50 = train_test_split(income_x, income_y, test_size=0.5)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(mushroom_x_train_50, mushroom_x_test_50, mushroom_y_train_50, mushroom_y_test_50, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_mushroom_accs.append(test_accuracy)


# In[119]:


# 80% Training and 20% Testing 
mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20 = train_test_split(income_x, income_y, test_size=0.2)
opt_D, cross_validation_accuracies, cross_validation_errors, mean_training_accuracies, mean_training_errors, test_accuracy, test_error  =  calcDTMetrics(mushroom_x_train_80, mushroom_x_test_20, mushroom_y_train_80, mushroom_y_test_20, D_list)

draw_heatmap(cross_validation_errors, D_list, title='cross-validation error w.r.t D')
print("Best D: {}".format(opt_D))
print("Test error: {}".format(test_error))
print("Test accuracy: {}".format(test_accuracy))
print("Avg training error per D: {}".format(mean_training_errors))
print("Avg training accuracies per D: {}".format(mean_training_accuracies))

dt_mushroom_accs.append(test_accuracy)


# In[120]:


average_dt_accuracy = np.sum([a + b + c for a, b, c in zip(dt_bank_accs, dt_income_accs, dt_mushroom_accs)]) / 9
print(average_dt_accuracy)


# In[132]:


data = {
    'SVM': [svm_bank_accs, svm_income_accs, svm_mushroom_accs],
    'KNN': [knn_bank_accs, knn_income_accs, knn_mushroom_accs],
    'Decision Tree': [dt_bank_accs, dt_income_accs, dt_mushroom_accs],
}



# Data preparation
datasets = ['Bank', 'Income', 'Mushroom']
classifiers = ['SVM', 'KNN', 'Decision Tree']

# Plotting
for i, dataset in enumerate(datasets):
    plt.figure(figsize=(10, 6))
    plt.title(f'Classifier Accuracies for {dataset} Dataset')
    plt.xlabel('Training Split')
    plt.ylabel('Accuracy')

    for classifier in classifiers:
        accs = data[classifier][i]
        splits = ['20/80', '50/50', '80/20']
        plt.plot(splits, accs, label=classifier)

    plt.legend()
    plt.show()


# In[163]:


bar_width = 0.2  # Width of each bar
bar_positions = np.arange(len(datasets))
colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))

for i, split in enumerate(['20/80', '50/50', '80/20']):
    plt.figure(figsize=(10, 6))
    plt.title(f'Classifier Accuracies for {split} Training Split')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.xticks(bar_positions, datasets)
    
    for j, (classifier, color) in enumerate(zip(classifiers, colors)):
        accs = [data[classifier][k][i] for k in range(len(datasets))]
        plt.bar(bar_positions + j * bar_width, accs, width=bar_width, label=classifier,  color=color)
        
        avg_acc = np.mean(accs)
        print("Average Accuracy for {}: {}".format(classifier, avg_acc))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=len(classifiers))
  
    plt.grid(True)
    plt.show()


# In[123]:


average_accuracies = [np.mean(np.array(data[classifier])) for classifier in classifiers]


df_data = pd.DataFrame({
    'Classifier': classifiers,
    'Average Accuracy': average_accuracies
})

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Average Accuracy', y='Classifier', data=df_data, palette='viridis', ci=None)
plt.title('Average Accuracies of Classifiers')
plt.xlabel('Average Accuracy')
plt.ylabel('Classifier')

# Add labels for each point
for i, v in enumerate(average_accuracies):
    plt.text(v + 0.01, i, f'{v:.3f}', color='black', va='center', fontweight='bold')

plt.show()


# In[124]:


# Convert accuracy data to a 2D NumPy array
heatmap_data = np.array([[data[classifier][i][j] for j in range(len(datasets))] for i in range(len(datasets))])

# Plotting the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest', aspect='auto')

plt.colorbar(label='Accuracy')
plt.title('Classifier Accuracies Heatmap')
plt.xlabel('Dataset')
plt.ylabel('Classifier')
plt.xticks(np.arange(len(datasets)), datasets)
plt.yticks(np.arange(len(classifiers)), classifiers)

plt.show()


# In[ ]:





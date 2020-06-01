"""
Created on Sun May  3 13:29:36 2020
@author: DESHMUKH
KNN classifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
pd.set_option('display.max_columns',None)

# =============================================================================
# Business Problem - Prepare a model for glass classification using KNN
# =============================================================================

glass = pd.read_csv("glass.csv")
glass.head()
glass.isnull().sum()
glass.info()
glass.shape

# Value count of classes in output variables
glass.groupby('Type').size()

# Summary
glass.describe()

# Nomalization of data
#glass_trans = normalize(glass.iloc[:,0:9])
                                                # OR # 
# Standardization of data
glass_trans = scale(glass.iloc[:,0:9])

# Giving Columns names.
glass_trans = pd.DataFrame(glass_trans, columns=glass.columns[:-1])

# Histogram
glass_trans.hist()

# Boxplot
glass_trans.boxplot(notch='True',patch_artist=True,grid=False);plt.xticks(fontsize=6)

# Pair plot
sns.pairplot(glass)

######################## - Spliting data in X and y - ########################

X = glass_trans
y = glass['Type']

##################### - Spliting data in train and test - ####################

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3,random_state = False)

##################### - Building KNN Classifier Model - ######################

# Considering K is as sqrt of total of X train
knn = KNeighborsClassifier(n_neighbors=11) # Intially take nearest odd value to square root of count of X train data as K
knn.fit(X_train, y_train)

# Predication 
pred = knn.predict(X_test)

# Confusion matrix
confusion_matrix = pd.crosstab(y_test,pred,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix, annot = True, cmap = 'Blues')

# Accuaracy
print(accuracy_score(y_test, pred)) # 0.68

# Classification Report
print(classification_report(y_test,pred))

########################### - Choosing a Prefect K Value - #########################

# Finding the K value by using error (In other example I use Accuarcy)
error_rate = []

# Calculating error for K values between 1 and 40
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# Graphical Representaion of K values
plt.figure(figsize=(12,6))
plt.plot(range(1,40), error_rate, 'o-',color = 'k',markerfacecolor='g', markersize=8)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.xticks(np.arange(1,40)) # 1-start,40-end,2-jump
plt.ylabel('Mean Error')
plt.grid()

# Selecting K values is 3 as a best fit value from above plot.

##################### - Building KNN Classifier Final Model - ######################

knn_final = KNeighborsClassifier(n_neighbors=3) 
knn_final.fit(X_train, y_train)

# Predication 
pred_final_train = knn_final.predict(X_train)
pred_final_test = knn_final.predict(X_test)

# Confusion matrix (test data)
confusion_matrix_f = pd.crosstab(y_test,pred_final_test,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix_f, annot = True, cmap = 'RdPu')

# Accuaracy
accuracy_score(y_train, pred_final_train) # 0.83
accuracy_score(y_test, pred_final_test)   # 0.72

# Classifiaction Report
print(classification_report(y_test,pred_final_test))

                 # ---------------------------------------------------- #
















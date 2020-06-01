"""
Created on Sun May  3 21:35:18 2020
@author: DESHMUKH
KNN classifier
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
pd.set_option('display.max_columns',None)

# ===================================================================================
# Business Problem - Implement a KNN model to classify the animals in to categories. 
# ===================================================================================

zoo = pd.read_csv("Zoo.csv")
zoo.isnull().sum()
zoo.head()
zoo = zoo.iloc[:,1:18]
zoo.info()

# Value count of classes in output variable
zoo.groupby('type').size()

# Histrogram
zoo.hist()

# Boxplot
zoo.boxplot(notch='True',patch_artist=True,grid=False);plt.xticks(fontsize=6)

# This data are not have to convert into Standerdized or Normalized form as it is only contain 0 and 1 (it is already unitless and have same scale)

# Pairplot
#sns.pairplot(zoo.iloc[:,:16])

########################### - Spliting data in X and y - ###########################

X = zoo.iloc[:,:16]
y = zoo.iloc[:,16]

######################## - Spliting data in train and test - #######################

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = False )

######################## - Building KNN Classifier Model - #########################

# Considering K is as nearst odd no. to sqrt of total of X train
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train,y_train)

# Predication
pred = knn.predict(X_test)

# Confusion matrix
confusion_matrix = pd.crosstab(y_test,pred,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix, annot = True, cmap = 'Blues')

# Accuracy
accuracy_score(y_test,pred) # 0.88

# Classification Report
print(classification_report(y_test,pred))

########################### - Choosing a Prefect K Value - #########################

# Finding the K value by using Accuracy (In other example I use error but it is BEST)
# Running KNN algorithm for 1 to 30 nearest neighbours(odd numbers) and 
# storing the accuracy values

acc = []
for i in range(1,30,1):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train,y_train)
    train_acc = np.mean(knn_i.predict(X_train)==y_train)
    test_acc = np.mean(knn_i.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
    
# Accuracy Elbow plot
plt.plot(np.arange(1,30,1),[i[0] for i in acc],"go-",mfc='k')
plt.plot(np.arange(1,30,1),[i[1] for i in acc],"ro-",mfc='k')
plt.title('Accuracy')
plt.xlabel('K Value')
plt.xticks(np.arange(1,30)) # 1-start,30-end
plt.legend(["train","test"])
plt.grid()  
    
# Selecting K values is 5 as a best fit value from above plot.

##################### - Building KNN Classifier Final Model - ######################
    
knn_final = KNeighborsClassifier(n_neighbors = 5)
knn_final.fit(X_train,y_train)

# Predication
pred_final_train = knn_final.predict(X_train)
pred_final_test = knn_final.predict(X_test)

# Confusion matrix (test data)
confusion_matrix_f = pd.crosstab(y_test,pred_final_test,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix_f, annot = True, cmap = 'Reds')

# Accuracy
accuracy_score(y_train,pred_final_train) # 0.96
accuracy_score(y_test,pred_final_test) # 0.96

# Classification Report
print(classification_report(y_test,pred_final_test))
    
    
                        # ---------------------------------------------------- #
    
    







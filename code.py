# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 

df = pd.read_csv(path, header=None)
# Overview of the data
# df.info()
# df.describe()


#Dividing the dataset set in train and test set and apply base logistic model
X= df.iloc[:,:-1]
y = df.iloc[:,-1]
# print(X)
# print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

#Baseline Logistic regression model
model = LogisticRegression()
model_lr = model.fit(X_train,y_train)
y_pred = model_lr.predict(X_test)
# Calculate accuracy , print out the Classification report and Confusion Matrix.
accuracy = model_lr.score(X_test,y_test)
print("Baseline LR model accuracy is ",accuracy)
print("==========================================")
# print(classification_report(y_test, y_pred))
# print("==========================================")
# print(confusion_matrix(y_test, y_pred))
# print("==========================================")

# Copy df in new variable df1
df1 = df.copy()
corr = df1.drop(57,1).corr()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
#print(upper)

# Remove Correlated features above 0.75 and then apply logistic model
columns_to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
#print(columns_to_drop)
df1.drop(columns_to_drop, axis=1, inplace=True)

# Split the new subset of data and fit the logistic model on training data

X= df1.iloc[:,:-1]
y = df1.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)


#Baseline Logistic regression model 1
model_1 = LogisticRegression()
model_1_lr = model_1.fit(X_train,y_train)
y_pred = model_1_lr.predict(X_test)

# Calculate accuracy , print out the Classification report and Confusion Matrix for new data
accuracy = model_1_lr.score(X_test,y_test)
print("Corelated features removed LR model accuracy is ",accuracy)
print("======================================================================")
# print(classification_report(y_test, y_pred))
# print("==========================================")
# print(confusion_matrix(y_test, y_pred))
# print("==========================================")

# Apply Chi Square and fit the logistic model on train data use df dataset
n_features = [15,20,25,30,35,40,45,50,55]
highest_accuracy = 0
for i in n_features:
    test = SelectKBest(score_func = chi2, k= i)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

    X_train_transformed = test.fit_transform(X_train,y_train)
    X_test_transformed  = test.transform(X_test)

    # Logistic regression model 2
    model_2 = LogisticRegression()
    model_2_lr = model_2.fit(X_train_transformed,y_train)
    accuracy = model_2_lr.score(X_test_transformed,y_test)
    if accuracy >= highest_accuracy:
        highest_accuracy = accuracy
        optimal_n = i


print("Chi2 Features selected LR model with", optimal_n," features the accuracy is ", highest_accuracy)
 

# Calculate accuracy , print out the Confusion Matrix 
# print(classification_report(y_test, y_pred))
# print("==========================================")
# print(confusion_matrix(y_test, y_pred))
# print("==========================================")  

# Apply Anova and fit the logistic model on train data use df dataset

n_features = [5,10,15,20,25,30,35,40,45,50,55]
highest_accuracy = 0
for i in n_features:
    test = SelectKBest(score_func = f_classif, k= i)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

    X_train_transformed = test.fit_transform(X_train,y_train)
    X_test_transformed  = test.transform(X_test)

    # Logistic regression model 3
    model_3 = LogisticRegression()
    model_3_lr = model_3.fit(X_train_transformed,y_train)
    accuracy = model_3_lr.score(X_test_transformed,y_test)
    if accuracy >= highest_accuracy:
        highest_accuracy = accuracy
        optimal_n = i


print("Annova Features selected LR model with", optimal_n," features the accuracy is ", accuracy)

# Calculate accuracy , print out the Confusion Matrix 
# print(classification_report(y_test, y_pred))
# print("==========================================")
# print(confusion_matrix(y_test, y_pred))
# print("==========================================")  


# Apply PCA and fit the logistic model on train data use df dataset
n_components = [5,10,15,20,25,30,35,40,45,50,55]
n=10
highest_accuracy = 0
for n in n_components:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

    pca = PCA(n_components = n)
    

    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.fit_transform(X_test)

    # Logistic regression model 4
    model_4 = LogisticRegression()
    model_4_lr = model_4.fit(X_train_transformed,y_train)
    
    pca_accuracy = model_4_lr.score(X_test_transformed,y_test)
    print("accuracy with ",n,"features as ",pca_accuracy)
    if pca_accuracy >= highest_accuracy:
        highest_accuracy = pca_accuracy
        optimal_n = i

print("PCA Features selected LR model with", optimal_n," features the accuracy is ", pca_accuracy)


# Calculate accuracy , print out the Confusion Matrix 


# Compare observed value and Predicted value





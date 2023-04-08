import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
data=pd.read_csv("C:/Users/Primasolv/Desktop/3-2 minor project/BC mini project/breast_cancer_dataframe.csv")
print(data.columns)
data=data.iloc[:,2:]
print(data.columns)

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
l=['mean area','area error','worst area']
for i in l:
    data=remove_outlier(data,i)
upper = data.corr().where(np.triu(np.ones(data.corr().shape),
                                      k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.97)]
print(to_drop)
print(data.columns)
data = data.drop(data[to_drop], axis=1)
plt.figure(figsize=(50,50))
y=data.target
x=data.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split
px_train,x_test,py_train,y_test=train_test_split(x, y, test_size=0.2, random_state=5)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
x_train, y_train = sm.fit_sample(px_train, py_train.ravel())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state= 5)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
pickle.dump(rf_classifier, open('model.pkl','wb'))

 # Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]))

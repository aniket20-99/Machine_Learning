import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"E:\Datasets\logit classification.csv")

X = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=51)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7,weights ="distance",p = 1,algorithm = "kd_tree")
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm) 

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac) 

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr) 

bias = classifier.score(X_train,y_train)
print('bias:',bias) 

vairance = classifier.score(X_test,y_test)
print("variance:", vairance)


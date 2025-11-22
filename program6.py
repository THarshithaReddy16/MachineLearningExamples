import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data=pd.DataFrame({
    'Feature1':[2,4,4,6,6,8,10,12],
    'Feature2':[4,2,4,2,6,6,8,10],
    'Label':['A','A','B','B','A','B','A','B']
    })
print("DataSet:\n",data)
x=data[['Feature1','Feature2']]
y=data['Label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("\n Confusion matrix:\n",confusion_matrix(y_test,y_pred))
print("\n Classification Report:\n",classification_report(y_test,y_pred))
print("\n Accuracy :",accuracy_score(y_test,y_pred))

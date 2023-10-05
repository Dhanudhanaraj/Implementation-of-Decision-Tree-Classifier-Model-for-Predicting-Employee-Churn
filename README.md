# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:

1. Hardware – PCs
  
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:DHANUMALYA.D 
Register Number:212222230030  

```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")

df.head()

df.info()

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

### Initial data set:

![Screenshot from 2023-10-05 10-18-10](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/fea3b466-daae-4806-ba82-49af803eb682)

### Data info:

![Screenshot from 2023-10-05 10-18-21](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/c045c1b8-efdc-42c1-825d-6105686c54b1)

### Optimization of null values:

![Screenshot from 2023-10-05 10-18-31](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/66778c72-ede1-428c-8ad2-1070872d3abc)

### Assignment of x and y values:

![Screenshot from 2023-10-05 10-18-39](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/a559ba89-fa60-498c-92e1-89604838cf05)


![Screenshot from 2023-10-05 10-18-49](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/7e98c895-a4ca-4b99-9579-782eca09c741)

### Converting string literals to numerical values using label encoder:

![Screenshot from 2023-10-05 10-18-54](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/d5f3a8f1-a067-46ca-b3c5-bc510a1f579f)

### Accuracy:

![Screenshot from 2023-10-05 10-19-07](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/2101ae36-1a99-4092-a788-9b08a9904073)

### Prediction:
![Screenshot from 2023-10-05 10-19-18](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119218812/fd75cb26-a0bd-4edd-9b06-a4c1342d08e1)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

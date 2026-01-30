import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
df = pd.read_csv("finaldataset.csv")

x = df[['Age_at_Marriage','Gender_Encoded','Education_Level_Encoded','Caste_Match_Encoded','Religion_Encoded','Parental_Approval_Encoded','Dowry_Exchanged_Encoded','Marital_Satisfaction_Encoded','Years_Since_Marriage','Spouse_Working_Encoded','Inter-Caste_Encoded','Inter-Religion_Encoded']]
y = df['Marriage_Type_Encoded']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=8)

print(x_train)
print(y_test)

model = LogisticRegression()

model.fit(x_train,y_train)
pre = model.predict(x_test)
score  = accuracy_score(y_test,pre)
print("accuracy score :",score*100)
print("confusion matrix :",confusion_matrix(y_test,pre))





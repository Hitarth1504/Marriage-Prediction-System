import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("marriage_data_india.csv")


print(df.isnull().sum())

df_label = df.copy()

le = LabelEncoder()

df_label['Marriage_Type_Encoded'] = le.fit_transform(df['Marriage_Type'])
df_label['Gender_Encoded'] = le.fit_transform(df['Gender'])
df_label['Education_Level_Encoded'] = le.fit_transform(df['Education_Level'])
df_label['Caste_Match_Encoded'] = le.fit_transform(df['Caste_Match'])
df_label['Religion_Encoded'] = le.fit_transform(df['Religion'])
df_label['Parental_Approval_Encoded'] = le.fit_transform(df['Parental_Approval'])
df_label['Dowry_Exchanged_Encoded'] = le.fit_transform(df['Dowry_Exchanged'])
df_label['Marital_Satisfaction_Encoded'] = le.fit_transform(df['Marital_Satisfaction'])
df_label['Divorce_Status_Encoded'] = le.fit_transform(df['Divorce_Status'])
df_label['Income_Level_Encoded'] = le.fit_transform(df['Income_Level'])
df_label['Spouse_Working_Encoded'] = le.fit_transform(df['Spouse_Working'])
df_label['Inter-Caste_Encoded'] = le.fit_transform(df['Inter-Caste'])
df_label['Inter-Religion_Encoded'] = le.fit_transform(df['Inter-Religion'])

df_label.to_csv("finaldataset.csv",index=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso

dataframe= pd.read_csv("D:\\insurance_price_detection\\insurance.csv")

# print(dataframe.head())
# print(dataframe.shape)
# print(dataframe.describe())
# print(dataframe.info())
# print(dataframe.isnull().sum())
# print(dataframe.isnull().sum().sum())

# print(dataframe["sex"].value_counts())
# print(dataframe["region"].value_counts())
# print(dataframe["smoker"].value_counts())


dataframe["sex"] = (dataframe["sex"]).map({"male":0, "female":1})
dataframe["smoker"] = (dataframe["smoker"]).map({"no":0, "yes":1})
dataframe["region"] = (dataframe["region"]).map({"northeast":0, "northwest":1, "southwest":2, "southeast":3})

x= dataframe.drop(["charges"], axis= 1)
y= dataframe["charges"]

# print(x.isnull().sum())

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.1, random_state=2)

# model= LinearRegression()

model= Lasso()

model.fit(x_train, y_train)

train_data_score= model.predict(x_train)
test_data_score= model.predict(x_test)

print(metrics.r2_score(y_train, train_data_score))
print(metrics.r2_score(y_test, test_data_score))


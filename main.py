import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("F:/PythonProjects/FutureSalesPrediction/advertising.csv")
#print(data.head())

#check whether dataset contains null values
#print (data.isnull().sum())

#visualize the relationship between the amount ospent on advertising on TV and units sold
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter (data_frame = data, x="Sales", y="TV", size="TV", trendline="ols")

figure.show()

#visualize relationship between the amount sepnt on newspaper adverts and units sold

figure = px.scatter(data_frame = data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols")

figure.show()

#visualize relatinship between radio adverts and units sold

figure = px.scatter(data_frame = data, x="Sales", y="Radio", size = "Radio", trendline = "ols")
figure.show()

#correlation of all columns with the sales column
correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

#training a machine learning model to predict future sales of a product.
#first split the data into training and test sets

x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42)

#train model to predict future sales
model = LinearRegression()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))

#input values based on features we have used to predict how mant units can be sold based
# on the amount spent on its advertising on various platforms

features = np.array([[230.1, 37.8,69.2]])
print(model.predict(features))
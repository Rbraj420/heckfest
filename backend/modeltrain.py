import numpy as np
import pandas as pd

# Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Importing machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, accuracy_score, classification_report

# Importing statistical libraries
import statsmodels.api as sm

# Load the data
data = pd.read_csv("clothes.csv")

# Sort the data by "Profit %" column in ascending order
highest_profit = data.sort_values(by="Profit %", ascending=False)

# Convert "Invoice Date" column to datetime format
highest_profit["Invoice Date"] = pd.to_datetime(highest_profit["Invoice Date"])
print(highest_profit)

# Get the count of units sold for each invoice date
# adidas_counts = highest_profit["Invoice Date"].value_counts()
# adidas_sell_data= highest_profit["Unit Sold"].value_counts()

# # Plot the bar chart
# figure = px.bar(x=adidas_counts.index, y=adidas_sell_data., title="Number of units sold per invoice date")
# figure.show()

# #now to show the relationship bw sell price and invoice date
# figure=  px.scatter(data_frame=data,x= "Invoice Date",y="No. of Items sold", trendline="ols",title="Relationship bw Occasion and Sales")
# figure.show()




dataframe= pd.DataFrame(data)
X= dataframe[["Invoice Date"]]
y= dataframe[["Unit Sold"]]
# print(dataframe)
plt.plot(dataframe['Unit Sold'],dataframe['Invoice Date'])
plt.ylabel("Invoice Date")
plt.xlabel("Unit sold")
plt.show()

# X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# model= LinearRegression()
# model.fit(X_train,y_train)
# #to check whether my model works or not
# print(model)
# y_pred= model.predict(X_test)
# accuracy= accuracy_score(y_test,y_pred)
# report= classification_report(y_test,y_pred)

# #user input 
# user_date= int(input("Enter the date on which you want. "))
# prediction= model.predict(user_date)
# print("Predicted Sells are :",prediction)


# codes for errors
# mse= mean_squared_error(y_test, y_pred)
# print(mse)
# mape= mean_absolute_percentage_error(y_pred,y_test)
# print("Mean absolute Percentage error:",mape)







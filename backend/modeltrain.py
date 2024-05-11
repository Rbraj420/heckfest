import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score,classification_report
import plotly.graph_objects as go
import statsmodels.api as sm

data = pd.read_csv("apple_products.csv")
print(data.isnull().sum())
print(data.head(10))

# to analyse the data
# highest_profit= data.sort_values(by=["Profit %"],ascending=False)

# # if you want to see what are the occasion dates on which max products are sold
# print(highest_profit["Product"])
# # now labeling the data(Plotting in the graph)
# adidas = highest_profit["Product"].value_counts()
# label= adidas.index
# counts= highest_profit["Invoice Date"]
# figure= px.bar(highest_profit,x= adidas,y=counts,title="Number of units sold")
# #to show graph
# figure.show()


# #now to show the relationship bw sell price and invoice date
# figure=  px.scatter(data_frame=data,x= "Invoice Date",y="No. of Items sold", trendline="ols",title="Relationship bw Occasion and Sales")
# figure.show()




# dataframe= pd.DataFrame(data)
# X= dataframe[["Invoice Date"]]
# y= dataframe[["Unit Sold"]]

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





 


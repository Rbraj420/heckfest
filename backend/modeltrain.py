import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

data = pd.read_csv("AdidasData.csv")
print(data.isnull().sum())
dataframe= pd.DataFrame(data)
X= dataframe[["Invoice Date"]]
y= dataframe[["Units Sold"]]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

model= LinearRegression()
model.fit(X_train,y_train)

print(model)

 


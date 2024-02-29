import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

data = pd.read_csv('diamonds.csv')
data.head()
data.info()
data.describe()


data.isnull().sum()
data["cut"].value_counts()
fig =px.bar(data,x=data["cut"],title="Quality Count")
fig.show()


data["color"].value_counts()
fig = px.bar(data, x="color", title="Quality Count")
fig.show()



data["clarity"].value_counts()
fig =px.bar(data,x=data["clarity"],title="Quality Count")
fig.show()

try:
    data = data.drop("Unnamed: 0", axis=1)
except KeyError:
    pass
figure = px.scatter(data_frame = data, x="carat",
                    y="price", size="depth", 
                    color= "cut", trendline="ols")
figure.show()
data["size"] = data["x"] * data["y"] * data["z"]
data
figure = px.scatter(data_frame = data, x="size",
                    y="price", size="size", 
                    color= "cut", trendline="ols")
figure.show()
fig = px.box(data, x="cut", 
             y="price", 
             color="color")
fig.show()
fig = px.box(data, 
             x="cut", 
             y="price", 
             color="clarity")
fig.show()
data["cut"] = data["cut"].map({"Ideal": 1,  "Premium": 2,  "Good": 3, "Very Good": 4, "Fair": 5})
data["color"] = data["color"].map({"D": 1,  "E": 2,  "F": 3,  "G": 4,  "H": 5,   "I":6,  "J":7})
data["clarity"] = data["clarity"].map({"SI1": 1, "SI2": 2, "VS1": 3, "VS2": 4, "VVS1": 5,   "VVS2":6,  "I1":7, "IF":8})
data.head()

data["clarity"].value_counts()
fig =px.bar(data,x=data["clarity"],title="Quality Count")
fig.show()
try:
    data = data.drop("Unnamed: 0", axis=1)
except KeyError:
    pass
figure = px.scatter(data_frame = data, x="carat",
                    y="price", size="depth", 
                    color= "cut", trendline="ols")
figure.show()
data["size"] = data["x"] * data["y"] * data["z"]
data
figure = px.scatter(data_frame = data, x="size",
                    y="price", size="size", 
                    color= "cut", trendline="ols")
figure.show()
fig = px.box(data, x="cut", 
             y="price", 
             color="color")
fig.show()
fig = px.box(data, 
             x="cut", 
             y="price", 
             color="clarity")
fig.show()
data["cut"] = data["cut"].map({"Ideal": 1,  "Premium": 2,  "Good": 3, "Very Good": 4, "Fair": 5})
data["color"] = data["color"].map({"D": 1,  "E": 2,  "F": 3,  "G": 4,  "H": 5,   "I":6,  "J":7})
data["clarity"] = data["clarity"].map({"SI1": 1, "SI2": 2, "VS1": 3, "VS2": 4, "VVS1": 5,   "VVS2":6,  "I1":7, "IF":8})
data.head()
data.describe()

from sklearn.model_selection import train_test_split
x = np.array(data[["carat", "cut","color","clarity", "size"]])
y = np.array(data[["price"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.20, 
                                                random_state=42)

features = ["carat", "cut", "color", "clarity", "size"]
target = "price"

x = data[features].values
y = data[target].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)



imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtrain_df = pd.DataFrame(xtrain, columns=features)
ytrain_df = pd.DataFrame(ytrain, columns=[target])

df = pd.concat([xtrain_df, ytrain_df], axis=1)


df = df.dropna()


xtrain = df[features]
ytrain = df[target]


df = pd.concat([xtrain, ytrain], axis=1)


correlation = data.corr()
print(correlation["price"].sort_values(ascending=False))
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
model.score(xtrain, ytrain)
a = float(input("Carat Size: "))
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
c = int(input("color type(D: 1 E: 2 F: 3 G: 4 H: 5 I:6 J:7):   "))
d = int(input("clarity type(SI1: 1  SI2: 2 VS1: 3 VS2: 4 VVS1: 5 VVS2:6 I1:7 IF:8):  "))
e = float(input("Size: "))
features = np.array([[a, b, c,d,e]])
pred = model.predict(features)
print("Predicted Diamond's Price($) = ", pred)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)
model.score(xtrain, ytrain)
a = float(input("Carat Size: "))
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
c = int(input("color type(D: 1 E: 2 F: 3 G: 4 H: 5 I:6 J:7):   "))
d = int(input("clarity type(SI1: 1  SI2: 2 VS1: 3 VS2: 4 VVS1: 5 VVS2:6 I1:7 IF:8):  "))
e = float(input("Size: "))
features = np.array([[a, b, c,d,e]])
pre = model.predict(features)
print("Predicted Diamond's Price ($) = ", pre)

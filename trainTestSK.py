import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/diamonds.csv.bz2")

X = df[["carat", "x", "y", "z"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
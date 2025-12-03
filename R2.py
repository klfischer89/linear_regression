import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score    
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./data/diamonds.csv.bz2")

xs = df["carat"].to_numpy().reshape(-1, 1)
ys = df["price"]

model = LinearRegression()
model.fit(xs, ys)

print(model.score(xs, ys))

y_pred = model.predict(xs)
print(r2_score(ys, y_pred))

xs = df[["carat", "x"]]
ys = df["price"]

model = LinearRegression()
model.fit(xs, ys)

# print(model.coef_)
# print(model.intercept_)

print(model.score(xs, ys))

print(model.predict([[5, 4]]))
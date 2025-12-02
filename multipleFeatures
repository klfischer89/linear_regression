import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./data/diamonds.csv.bz2")

model = LinearRegression()

X_train = df[["carat", "x"]]
y_train = df["price"]

model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

X_pred = np.array([
    [0.5, 5],
    [0.25, 3]
])

y_pred = model.predict(X_pred)
print(y_pred)
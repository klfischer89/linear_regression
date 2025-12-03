import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score    
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./data/diamonds.csv.bz2")

df = df.sample(frac = 1)

df_train = df.iloc[:40000]
df_test = df.iloc[40000:]

X_train = df_train["carat"].to_numpy().reshape(-1, 1)
y_train = df_train["price"]

model = LinearRegression()
model.fit(X_train, y_train)

X_test = df_test["carat"].to_numpy().reshape(-1, 1)
y_test = df_test["price"]

print(model.score(X_test, y_test))
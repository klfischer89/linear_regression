import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./data/diamonds.csv.bz2")

# sns.scatterplot(x = "carat", y = "price", data = df.sample(50))
# plt.show()

xs = df["carat"].to_numpy().reshape(-1, 1)
ys = df["price"].to_numpy()

model = LinearRegression()
model.fit(xs, ys)

# print(model.coef_)
# print(model.intercept_)

model.predict(np.array([
    [10],
    [1]
]))

x_pred = np.array([3, 0])
y_pred = model.predict(x_pred.reshape(-1, 1))

ax = sns.lineplot(x = x_pred, y = y_pred, color = "red")
sns.scatterplot(x = "carat", y = "price", data = df.sample(50), ax = ax)
plt.show()
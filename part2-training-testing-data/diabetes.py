import numpy as np
import matplotlib.pylot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
y = data.target.values
data = data.frame 

x = data["bmi"].values

x = x.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x,y test_size = .10)

model = LinearRegression().fit(xtrain,ytrain)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

print("Model's Linear Equation: y=", coef, "x+", intercept)
print("R Squared value:", r_squared)

predict = model.predict(xtest)
predict = np.around(predict, 2)



print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    print("x value:", float(x_coord), "Predicted y value:", predicted_y, "Actual y value:", actual)




plt.figure(figsize=(6, 4))

plt.scatter(xtrain,ytrain, c="blue")
plt.scatter(xtest, ytest, c="red")
plt.plot(xtrain, coef*xtrain + intercept, c="r", label="Line of Best Fit")

plt.xlabel("bmi")
plt.ylabel("quantitative measure")
plt.title("quantitative measure vs bmi")

plt.legend()
plt.show()
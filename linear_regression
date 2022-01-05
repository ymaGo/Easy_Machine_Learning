import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

experiences = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salaries = np.array([103100, 104900, 106800, 108700, 110400, 112300, 114200, 116100, 117800, 119700, 121600])

# transform the experiences to on column
experiences = experiences.reshape(-1, 1)

# creat liner regression model
regr = linear_model.LinearRegression()

# train
regr.fit(experiences, salaries)

print(regr.coef_)
print(regr.intercept_)

# predict by the trained model
diabetes_y_pred = regr.predict(experiences)

# plot the test result
plt.scatter(experiences, salaries, color='black')
plt.plot(experiences, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

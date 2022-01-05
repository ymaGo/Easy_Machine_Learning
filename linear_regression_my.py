import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

experiences = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salaries = np.array([103100, 104900, 106800, 108700, 110400, 112300, 114200, 116100, 117800, 119700, 121600])

#转换成1列，(1,-1)是1行
experiences = experiences.reshape(-1, 1)

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型——看就这么简单，一行搞定训练过程
regr.fit(experiences, salaries)

print(regr.coef_)
print(regr.intercept_)

# 用训练得出的模型进行计算
diabetes_y_pred = regr.predict(experiences)

# 将测试结果以图标的方式显示出来
plt.scatter(experiences, salaries, color='black')
plt.plot(experiences, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

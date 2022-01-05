from sklearn.linear_model import LogisticRegression
import pandas as pd

# Importing dataset
data = pd.read_csv('score.csv', delimiter=',')

used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

# Logistic Regression – Binary Classification
passed = []

for i in range(len(scores)):
    if (scores[i] >= 60):
        passed.append(1)
    else:
        passed.append(0)

classifier = LogisticRegression(solver='lbfgs', C=1e5)
classifier.fit(X, passed)

print(classifier.intercept_)
print(classifier.coef_)

# [-219.62853047]
# [[2.32811121 0.76351894]]

# I also calculated by Excel, but the results are different, I don't know the reason now!
# excel -127.8557416	1.464310039	   0.382675646

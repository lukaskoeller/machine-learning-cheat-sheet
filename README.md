# machine-learning-cheat-sheet

## Training & Test Data
```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 51)
```

## Linear Regression
```py
mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

score_training = mlr.score(x_train, y_train)
print(score_training)

score_test = mlr.score(x_test, y_test)
print(score_test)
```

## K-Nearest Neighbor Regressor
```py
from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")

regressor.fit(movie_dataset, movie_ratings)

print(regressor.predict([[0.016, 0.300, 1.022], [0.0004092981, 0.283, 1.0112], [0.00687649, 0.235, 1.0112]]))
```

## Normalization

## Logistic Regression
```py
from sklearn.linear_model import LogisticRegression
cc_lr = LogisticRegression()
cc_lr.fit(X_train,y_train)

result = cc_lr.predict(X_test) # 0 | 1
result_proba = cc_lr.predict_proba(X_test)
```

### Confusion Matrix
Shows the number of true positives, false positives, true negatives, and false negatives.
```py
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred))
```

### Accuracy, Recall, Precision, F1 Score
- Accuracy = (TP + TN)/(TP + FP + TN + FN)
- Precision = TP/(TP + FP)
- Recall = TP/(TP + FN)
- F1 score: weighted average of precision and recall
```py
# accuracy:
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))
# output: 0.7

# precision:
from sklearn.metrics import precision_score
print(precision_score(y_true, y_pred))
# output: 0.67

# recall: 
from sklearn.metrics import recall_score
print(recall_score(y_true, y_pred))
# output: 0.8

# F1 score
from sklearn.metrics import f1_score
print(f1_score(y_true, y_pred))
# output: 0.73
```

## Glossary
- **Sigmoid Function**
- **Classification Thresholding**
- **Imbalanced class classification problem**

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

x, y = mnist["data"], mnist["target"]

x.shape

y.shape

some_digit = x[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

y = y.astype(np.uint8)

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train)

sgd_clf.predict([some_digit])

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct / len(y_pred))

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, x_train, y_train, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(sgd_clf, x_train, y_train, cv=3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_train_predict)

# from sklearn.metrics import precision_score, recall_score, f1_score

# precision_score(y_train, y_train_predict)
# recall_score(y_train, y_train_predict)
# f1_score(y_train, y_train_predict)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_score)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1], [0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()
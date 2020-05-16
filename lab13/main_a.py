import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from data import get_data
import matplotlib.pyplot as plt


def get_acc(neighbors):
    knc = KNeighborsClassifier(neighbors)
    data = get_data('glass.csv', 0.15, True)
    knc.fit(data['train']['data'], data['train']['target'])
    y_test_pred = knc.predict(data['test']['data'])
    test_acc = (data['test']['target'] == y_test_pred).sum() / data['test']['data'].shape[0]
    return test_acc


X = np.arange(1, 100)
Y = [get_acc(x) for x in X]

plt.ylim([0, 1])
plt.plot(X, Y)
plt.title('Стекло')
plt.xlabel('Количество соседей')
plt.ylabel('Точность')

plt.show()

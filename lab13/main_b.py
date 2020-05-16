from sklearn.neighbors import KNeighborsClassifier
from data import get_data
import matplotlib.pyplot as plt

data = get_data('glass.csv', 0.15, True)


def get_acc(metric):
    knc = KNeighborsClassifier(n_neighbors=4, metric=metric)
    knc.fit(data['train']['data'], data['train']['target'])
    y_test_pred = knc.predict(data['test']['data'])
    test_acc = (data['test']['target'] == y_test_pred).sum() / data['test']['data'].shape[0]
    return test_acc


metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
acc = [get_acc(m) for m in metrics]

fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(111)
ax.bar(metrics, acc)
ax.set_ylim([0, 1])
ax.set_xlabel('Метрика')
ax.set_ylabel('Точность')
plt.title('Зависимость точности от выбранной метрики')
plt.show()

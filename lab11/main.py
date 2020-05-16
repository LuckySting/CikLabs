from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from data import load_ttt_data, load_spam_data


def get_acc(test_p, loader, filename):
    data = loader(filename, test_p, shuffle=True)
    gnb = GaussianNB()
    gnb.fit(data['train']['data'], data['train']['target'])
    y_test_pred = gnb.predict(data['test']['data'])
    y_train_pred = gnb.predict(data['train']['data'])
    test_acc = (data['test']['target'] == y_test_pred).sum() / data['test']['data'].shape[0]
    train_acc = (data['train']['target'] == y_train_pred).sum() / data['train']['data'].shape[0]
    return train_acc, test_acc


X = np.linspace(0.01, 0.99, 100)

D_ttt = np.array([get_acc(x, load_ttt_data, 'tic_tac_toe.txt') for x in X])
D_spam = np.array([get_acc(x, load_spam_data, 'spam.csv') for x in X])

train_ttt = D_ttt[:, 0]
test_ttt = D_ttt[:, 1]

train_spam = D_ttt[:, 0]
test_spam = D_ttt[:, 1]

plt.subplot(121)
plt.plot(X, train_ttt, X, test_ttt)
plt.legend(('На обучающей выборке', 'На тестовой выборке'))
plt.title('Крестики-нолики')
plt.xlabel('Часть тестовых данных')
plt.ylabel('Точность')

plt.subplot(122)
plt.plot(X, train_spam, X, test_spam)
plt.legend(('На обучающей выборке', 'На тестовой выборке'))
plt.title('Спам')
plt.xlabel('Часть тестовых данных')
plt.ylabel('Точность')

plt.show()

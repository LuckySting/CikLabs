import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from data import get_data


def main(find_good=False, target_accuracy=0.95):
    """
    Выполняет задание на нахождение оптимального Байесовского классификатора
    :param find_good: Найти "хороший" классификатор?
    :param target_accuracy: Целевая точность
    """

    while True:
        data = get_data(0.2, True)
        c1_train_i = [data['train']['target'][i] == 1 for i in range(len(data['train']['target']))]
        c1_train = data['train']['data'][c1_train_i]
        c2_train = data['train']['data'][np.logical_not(c1_train_i)]
        c1_test_i = [data['test']['target'][i] == 1 for i in range(len(data['test']['target']))]
        c1_test = data['test']['data'][c1_test_i]
        c2_test = data['test']['data'][np.logical_not(c1_test_i)]

        gnb = GaussianNB()
        gnb.fit(data['train']['data'], data['train']['target'])
        y_test_pred = gnb.predict(data['test']['data'])

        test_acc = (data['test']['target'] == y_test_pred).sum() / data['test']['data'].shape[0]
        acc_matrix = confusion_matrix(data['test']['target'], y_test_pred)
        roc_fpr, roc_tpr, _ = roc_curve(data['test']['target'], y_test_pred)
        pr_pres, pr_rec, _ = precision_recall_curve(data['test']['target'], y_test_pred)

        if not find_good or (test_acc >= target_accuracy and acc_matrix[0][1] + acc_matrix[1][0] < 2):
            break


    x_max = np.max(np.concatenate((c1_train[:, 0], c2_train[:, 0], c1_test[:, 0], c2_test[:, 0]))) + 1
    y_max = np.max(np.concatenate((c1_train[:, 1], c2_train[:, 1], c1_test[:, 1], c2_test[:, 1]))) + 1
    x_min = np.min(np.concatenate((c1_train[:, 0], c2_train[:, 0], c1_test[:, 0], c2_test[:, 0]))) - 1
    y_min = np.min(np.concatenate((c1_train[:, 1], c2_train[:, 1], c1_test[:, 1], c2_test[:, 1]))) - 1

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.scatter(c1_train[:, 0], c1_train[:, 1], c='r', vmax=x_max)
    ax1.scatter(c2_train[:, 0], c2_train[:, 1], c='b', vmax=x_max)
    plt.title('Обучающие данные')

    ax2 = fig.add_subplot(222)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.scatter(c1_test[:, 0], c1_test[:, 1], c='r', vmax=x_max)
    ax2.scatter(c2_test[:, 0], c2_test[:, 1], c='b', vmax=x_max)
    plt.title('Тестовые данные')

    ax3 = fig.add_subplot(223)
    ax3.set_xlim([-0.01, 1.01])
    ax3.set_xlabel('False positive rate')
    ax3.set_ylim([-0.01, 1.01])
    ax3.set_ylabel('True positive rate')
    ax3.plot(roc_fpr, roc_tpr, 'red')
    ax3.plot([0, 0, 1], [0, 1, 1], 'b--')
    ax3.plot([0, 1], [0, 1], 'c-.')
    ax3.legend(['Исследуемый', 'Идеальный', 'Случайный'])
    plt.title('ROC кривая')

    ax4 = fig.add_subplot(224)
    ax4.set_xlim([-0.01, 1.01])
    ax4.set_xlabel('Recall')
    ax4.set_ylim([-0.01, 1.01])
    ax4.set_ylabel('Precision')
    ax4.plot(pr_rec, pr_pres, 'orange')
    ax4.plot([0, 1, 1], [1, 1, 0], 'b--')
    ax4.plot([0, 1], [0, 0], 'c-.')
    ax4.legend(['Исследуемый', 'Идеальная', 'Случайный'])
    plt.title('PR кривая')

    plt.show()

    print('Точность: {}'.format(test_acc))
    print('Матрица ошибок:\n{}'.format(acc_matrix))
    if test_acc >= target_accuracy:
        if target_accuracy == 1:
            print('Вывод: классификатор идеальный')
        else:
            print('Вывод: классификатор хороший')
    else:
        print('Вывод: классификатор плохой')


main(find_good=True, target_accuracy=1)

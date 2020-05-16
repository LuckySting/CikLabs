import numpy as np
import csv


# noinspection PyTypeChecker
def shuffle_in_unison_scary(a, b):
    """
    Перемешивает массивы в унисон без копирования
    :param a: первый массив
    :param b: второй массив
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def get_data(filename, test_p=0.001, shuffle=False):
    """
    Собирает датасет из файла
    :param filename: имя файла с данными
    :param shuffle: перемешать датасет?
    :param test_p: доля тестовых данных, число от 0 до 1
    :return: раделенный датасет
    """

    data = []
    target = []
    with open(filename, 'r') as file:
        r = 0
        for row in csv.reader(file):
            r += 1
            if r == 1:
                continue
            cl = int(row[-1])
            params = row[1:-1]
            data.append(params)
            target.append(cl)

    data = np.array(data)
    target = np.array(target)

    if shuffle:
        shuffle_in_unison_scary(data, target)

    test_size = int(data.shape[0] * test_p)

    data_test = data[:test_size, :]
    target_test = target[:test_size]
    data_train = data[test_size:]
    target_train = target[test_size:]

    return {
        'train': {
            'data': data_train.astype('float32'),
            'target': target_train.astype('float32')
        },
        'test': {
            'data': data_test.astype('float32'),
            'target': target_test.astype('float32')
        }
    }

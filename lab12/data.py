import math

import numpy as np


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


def get_data(test_p=0.001, shuffle=False):
    """
    Генерирует датасет из нормально распределенных точек
    :param shuffle: перемешать датасет?
    :param test_p: доля тестовых данных, число от 0 до 1
    :return: раделенный датасет
    """
    c_1 = np.random.normal([17, 16], [2, 2], size=(30, 2))
    c_2 = np.random.normal([16, 14], [math.sqrt(5), math.sqrt(5)], size=(70, 2))

    data = np.concatenate((c_1, c_2))
    target = np.concatenate(([1 for _ in c_1], [-1 for _ in c_2]))

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

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


def split_data(data, test_p, shuffle):
    """
    Разделяет датасет на тестовую и обучающую часть и перемешивает
    :param data: датасет
    :param test_p: какую часть сделать тестовой
    :param shuffle: перемешать
    :return: раделенный датасет
    """
    pos_test_size = int(len(data[1]) * test_p)
    neg_test_size = int(len(data[-1]) * test_p)
    positives_data_test = np.array(data[1][:pos_test_size])
    negatives_data_test = np.array(data[-1][:neg_test_size])
    positives_data_train = np.array(data[1][pos_test_size:])
    negatives_data_train = np.array(data[-1][neg_test_size:])

    data_train = np.concatenate((positives_data_train, negatives_data_train))
    target_train = np.concatenate(([1 for _ in positives_data_train], [-1 for _ in negatives_data_train]))
    data_test = np.concatenate((positives_data_test, negatives_data_test))
    target_test = np.concatenate(([1 for _ in positives_data_test], [-1 for _ in negatives_data_test]))

    if shuffle:
        shuffle_in_unison_scary(data_train, target_train)
        shuffle_in_unison_scary(data_test, target_test)

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


def load_ttt_data(filename, test_p=0.001, shuffle=True):
    """
    Загружает данные для классификатора крестики нолики
    :param shuffle: перемешать массивы?
    :param filename: имя датасета
    :param test_p: доля тестовых данных, число от 0 до 1
    :return: раделенный датасет
    """
    data = dict()
    with open(filename) as file:
        for row in file:
            cl = 1 if 'pos' in row.split(',')[-1] else -1
            params = [-1 if s == 'x' else 1 for s in row.split(',')[:-1]]
            if cl not in data:
                data[cl] = [params]
            else:
                data[cl].append(params)

    return split_data(data, test_p, shuffle)


def load_spam_data(filename, test_p=0.0, shuffle=True):
    """
    Загружает данные для классификатора спама
    :param shuffle: перемешать массивы?
    :param filename: имя датасета
    :param test_p: доля тестовых данных, число от 0 до 1
    :return: раделенный датасет
    """
    data = dict()
    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        r = 0
        for row in reader:
            if r == 0:
                r += 1
                continue
            cl = -1 if 'non' in row[-1] else 1
            params = row[1:-1]
            if cl not in data:
                data[cl] = [params]
            else:
                data[cl].append(params)
    return split_data(data, test_p, shuffle)

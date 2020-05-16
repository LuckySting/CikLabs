import numpy as np
from keras.layers import Dense
from keras.models import *
from keras.optimizers import Adam

from lab21.data import get_data


def get_model():
    inp = Input((2,))
    dns = Dense(1, activation='sigmoid')(inp)
    model = Model(input=inp, output=dns)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_model2():
    inp = Input((2,))
    dns1 = Dense(2, activation='relu')(inp)
    dns2 = Dense(1, activation='sigmoid')(dns1)
    model = Model(input=inp, output=dns2)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


data = get_data('nn_1.csv', 0.05, True)
model = get_model2()
model.fit(data['train']['data'], data['train']['target'], steps_per_epoch=100, epochs=500)

predict = model.predict(data['test']['data'])
print(predict.flatten())
print(data['test']['target'])

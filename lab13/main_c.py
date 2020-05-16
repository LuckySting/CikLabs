import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from data import get_data
import matplotlib.pyplot as plt


data = get_data('glass.csv', 0, True)
knc = KNeighborsClassifier(n_neighbors=4, metric='manhattan')
knc.fit(data['train']['data'], data['train']['target'])

inp = np.array([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])

predict = knc.predict(inp)

print('Predicted type is:\n{}'.format(int(predict[0])))
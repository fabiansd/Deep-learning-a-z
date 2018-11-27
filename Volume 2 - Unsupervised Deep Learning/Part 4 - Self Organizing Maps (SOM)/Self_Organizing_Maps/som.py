# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
# show()
# Finding the frauds
mappings = som.win_map(X)

list_of_winning_nodes = []
while True:
    i = input('write a two dimensional touple. Write x to when finished\n ')
    i = tuple(i)
    list_of_winning_nodes.append(i)
    break
print(list_of_winning_nodes)
    # if isinstance(i,tuple) and len(i) == 2:
    #     list_of_winning_nodes.append(i)
    # elif i == 'x':
    #     break
    # else:
    #     print('please write a two-dimensional tuple')

    # if len(list_of_winning_nodes) > 9:
    #     break

frauds = np.array([])
for x in list_of_winning_nodes:
    frauds = np.concatenate([frauds, mappings[x]])

# frauds = np.concatenate((mappings[(8,2)],mappings[(2,2)]), axis = 0)
# frauds = np.array((mappings[(3,6)]))

print('Frauds array shape: {}'.format(frauds.shape))
frauds = sc.inverse_transform(frauds)

print(frauds)
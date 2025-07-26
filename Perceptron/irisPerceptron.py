import numpy as np
import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf8')
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
X = df.iloc[0:100, [0, 2]].values
""" plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('Sepal length (cm)')
plt.ylabel("Petal length (cm)")
plt.legend(loc='upper left')
plt.savefig('sapelVSpetal.png') """
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1),ppn.errors_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.savefig("EpochsVSUpdates")
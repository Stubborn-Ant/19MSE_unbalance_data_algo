from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.3, 0.7],
                           class_sep=0.8, random_state=0)
import numpy as np
import matplotlib.pyplot as plt
N = 1000
x = np.random.randn(N)
y = np.random.randn(N)
plt.scatter(x, y)
plt.show()
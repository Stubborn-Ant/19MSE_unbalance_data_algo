# -*- encoding:utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, ADASYN

print(__doc__)


def plot_resampling(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

    return c0, c1


if __name__ == '__main__':
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.3, 0.7],
                               n_informative=3, n_redundant=1, flip_y=0,
                               n_features=20, n_clusters_per_class=1, n_samples=80, random_state=10)
    ##使用PCA降维到两维,方便进行可视化
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X)

    ###运用SMOTE算法
    kind = ['regular', 'borderline1', 'borderline2', 'svm']
    sm = [SMOTE(k) for k in kind]
    X_resampled = []
    y_resampled = []
    X_res_vis = []
    for method in sm:
        X_res, y_res = method.fit_sample(X, y)
        X_resampled.append(X_res)
        y_resampled.append(y_res)
        X_res_vis.append(pca.fit_transform(X_res))

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    ##展示结果
    ax2.axis('off')
    ax_res = [ax3, ax4, ax5, ax6]

    c0, c1 = plot_resampling(ax1, X_vis, y, 'original_set')

    for i in range(len(kind)):
        plot_resampling(ax_res[i], X_res_vis[i], y_resampled[i], 'smote{}'.format(kind[i]))

    ax2.legend((c0, c1), ('Class #0', 'Class #1'), loc='center', ncol=1, labelspacing=0.)
    plt.tight_layout()
    plt.show()

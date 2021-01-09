from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(555)

# 使用make_classification 制造数据
# 模仿论文所使用的数据集 2%:98% 为了可视化方便使用2D数据
X, y = make_classification(n_samples=500,
                           n_features=9,
                           n_redundant=3,
                           weights=(0.1, 0.9),
                           n_clusters_per_class=2,
                           random_state=95)


# 15尤佳
# X.shape, y.shape
# y = []
# for x1 in X:
#     if x1[0] > 2:
#         y.append(0)
#     else:
#         y.append(1)
# y = np.array(y)


def view_y(y):
    print(f"class 0: {len(np.where(y == 0)[0])}\nclass 1: {len(np.where(y == 1)[0])}")


view_y(y)

plt.figure(figsize=(10, 8))


def plot2class(pos, neg, title=""):
    plt.scatter(pos[:, 0], pos[:, 1], marker='+',
                c="#ff0000", label="Positive")

    plt.scatter(neg[:, 0], neg[:, 1], marker='.',
                c="#3D9140", label="Negative")
    plt.legend()
    plt.title(title)
    plt.xlabel("Attrs 1")
    plt.ylabel("Attrs 2")


plot2class(X[y == 0], X[y == 1], "original dataset")
plt.show()


def NaiveSMOTE(X, N=100, K=5):
    """
    {X}: minority class samples;
    {N}: Amount of SMOTE; default 100;
    {K} Number of nearest; default 5;
    """
    # {T}: Number of minority class samples;
    T = X.shape[0]
    if N < 100:
        T = (N / 100) * T
        N = 100
    N = (int)(N / 100)

    numattrs = X.shape[1]
    samples = X[:T]
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(samples)
    Synthetic = np.zeros((T * N, numattrs))
    newindex = 0

    def Populate(N, i, nns, newindex):
        """
        Function to generate the synthetic samples.
        """
        for n in range(N):
            nn = np.random.randint(0, K)
            for attr in range(numattrs):
                dif = samples[nns[nn], attr] - samples[i, attr]
                gap = np.random.random()
                Synthetic[newindex, attr] = samples[i, attr] + gap * dif
            newindex += 1
        return newindex

    for i in range(T):
        nns = neigh.kneighbors([samples[i]], K, return_distance=False)
        newindex = Populate(N, i, nns[0], newindex)
    return Synthetic


X_over_sampling = NaiveSMOTE(X[y == 0], N=800)

X_over_sampling.shape

new_X = np.r_[X, X_over_sampling]
new_y = np.r_[y, np.zeros((X_over_sampling.shape[0]))]
new_X.shape, new_y.shape

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plot2class(X[y == 0], X[y == 1],
           title="original datasets")

plt.subplot(1, 2, 2)
plot2class(new_X[new_y == 0], new_X[new_y == 1],
           title="Naive SMOTE datasets")
plt.show()

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=95)
X_res, y_res = sm.fit_resample(X, y)

print("thrid SMOTE")
view_y(y_res)
print("Naive SMOTE")
view_y(new_y)

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plot2class(new_X[new_y == 0], new_X[new_y == 1],
           title="Naive SMOTE data sets")

plt.subplot(1, 2, 2)
plot2class(X_res[y_res == 0], X_res[y_res == 1],
           title="third SMOTE data sets")
plt.show()

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, plot_roc_curve

# 使用决策树, Gaussian Navie Bayes验证元数据与SMOTE采样后的数据的表现
# 测试元数据
stkf = StratifiedKFold(n_splits=5, random_state=95, shuffle=True)
fold = 1

ori_X, ori_X_test, \
ori_y, ori_y_test = train_test_split(X, y, random_state=95)

best_tree_est = None
best_tree_f1 = -np.Inf
best_gnb_est = None
best_gnb_f1 = -np.Inf
best_svc_f1 = -np.Inf

#################################ROC
from sklearn import svm

for train_index, vaild_index in stkf.split(ori_X, ori_y):
    X_train, y_train = ori_X[train_index], ori_y[train_index]
    X_test, y_test = ori_X[vaild_index], ori_y[vaild_index]

    tree = DecisionTreeClassifier(random_state=95)
    tree.fit(X_train, y_train)
    ## SMOTE算法
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    weights = np.where(y_train == 0, 9, 1)
    # svc = svm.SVC(kernel='sigmoid', gamma=2.0, tol=0.01)
    # svc.fit(X_train, y_train, sample_weight=weights)

    svc = RandomForestClassifier()
    svc.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)
    y_pred_gnb = gnb.predict(X_test)
    y_pred_svc = svc.predict(X_test)

    y_f1score_tree = f1_score(y_test, y_pred_tree)
    y_f1score_gnb = f1_score(y_test, y_pred_gnb)
    y_f1score_svc = f1_score(y_test, y_pred_svc)

    if y_f1score_tree > best_tree_f1:
        best_tree_f1 = y_f1score_tree
        best_tree_est = tree

    if y_f1score_gnb > best_gnb_f1:
        best_gnb_f1 = y_f1score_gnb
        best_gnb_est = gnb

    if y_f1score_svc > best_svc_f1:
        best_svc_f1 = y_f1score_svc
        best_svc_est = svc

    print(f"Fold {fold}:\nTree f1_score: {y_f1score_tree}\tTree Best\
     f1_score: {best_tree_f1}\nGNB f1_score: {y_f1score_gnb}\tGNB Best \
     f1_score: {best_gnb_f1}\n{40 * '-'}")

fig = plt.figure(figsize=(10, 8))
ax = plt.gca()
fig.add_axes(ax)
plt.title("Original dataset ROC AUC")
plot_roc_curve(best_tree_est, ori_X_test, ori_y_test, name="DTree", ax=ax)
plot_roc_curve(best_gnb_est, ori_X_test, ori_y_test, name="GNB", ax=ax)
plot_roc_curve(best_svc_est, ori_X_test, ori_y_test, name="RF", ax=ax)
plt.show()
###############################################  smote ###############################################
# 现在测试我们自己的SMOTE的数据
stkf = StratifiedKFold(n_splits=5, random_state=95, shuffle=True)
fold = 1

best_tree_est = None
best_tree_f1 = -np.Inf
best_gnb_est = None
best_gnb_f1 = -np.Inf
best_rf_f1 = -np.Inf

my_smote_X, my_smote_X_test, \
my_smote_y, my_smote_y_test = train_test_split(new_X, new_y,
                                               random_state=95)

for train_index, vaild_index in stkf.split(my_smote_X, my_smote_y):
    X_train, y_train = my_smote_X[train_index], my_smote_y[train_index]
    X_test, y_test = my_smote_X[vaild_index], my_smote_y[vaild_index]

    tree = DecisionTreeClassifier(random_state=95)
    tree.fit(X_train, y_train)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    rf=RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)
    y_pred_gnb = gnb.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    y_f1score_tree = f1_score(y_test, y_pred_tree)
    y_f1score_gnb = f1_score(y_test, y_pred_gnb)
    y_f1score_rf = f1_score(y_test, y_pred_rf)

    if y_f1score_tree > best_tree_f1:
        best_tree_f1 = y_f1score_tree
        best_tree_est = tree

    if y_f1score_gnb > best_gnb_f1:
        best_gnb_f1 = y_f1score_gnb
        best_gnb_est = gnb
    if y_f1score_rf > best_rf_f1:
        best_rf_f1 = y_f1score_rf
        best_rf_est = rf
    print(f"Fold {fold}:\nTree f1_score: {y_f1score_tree}\tTree Best\
     f1_score: {best_tree_f1}\nGNB f1_score: {y_f1score_gnb}\tGNB Best \
     f1_score: {best_gnb_f1}\n{40 * '-'}")

fig = plt.figure(figsize=(10, 8))
ax = plt.gca()
fig.add_axes(ax)
plt.title("Naive SMOTE dataset")
plot_roc_curve(best_tree_est, my_smote_X_test, my_smote_y_test, name="DTree", ax=ax)
plot_roc_curve(best_gnb_est, my_smote_X_test, my_smote_y_test, name="GNB", ax=ax)
plot_roc_curve(best_rf_est, my_smote_X_test, my_smote_y_test, name="RF", ax=ax)
plt.show()

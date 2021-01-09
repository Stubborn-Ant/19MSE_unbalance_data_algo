from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.sampling import cost_sampling, undersampling
from costcla.metrics import savings_score

data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets

X_cps_o, y_cps_o, cost_mat_cps_o = cost_sampling(X_train, y_train, cost_mat_train, method='OverSampling')
X_cps_r, y_cps_r, cost_mat_cps_r = cost_sampling(X_train, y_train, cost_mat_train, method='RejectionSampling')
X_u, y_u, cost_mat_u = undersampling(X_train, y_train, cost_mat_train)

#常规
y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
#过采样
y_pred_test_rf_cps_o = RandomForestClassifier(random_state=0).fit(X_cps_o, y_cps_o).predict(X_test)
#拒采样
y_pred_test_rf_cps_r = RandomForestClassifier(random_state=0).fit(X_cps_r, y_cps_r).predict(X_test)
#欠采样
y_pred_test_rf_u = RandomForestClassifier(random_state=0).fit(X_u, y_u).predict(X_test)
# Savings using only RandomForest
print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
# 0.12454256594
# Savings using RandomForest with cost-proportionate over-sampling
print(savings_score(y_test, y_pred_test_rf_cps_o, cost_mat_test))
# 0.192480226286
# Savings using RandomForest with cost-proportionate rejection-sampling
print(savings_score(y_test, y_pred_test_rf_cps_r, cost_mat_test))
# 0.465830173459
# Savings using RandomForest with under-sampling
print(savings_score(y_test, y_pred_test_rf_u, cost_mat_test))
# 0.466630646543
# Size of each training set
print(X_train.shape[0], X_cps_o.shape[0], X_cps_r.shape[0], X_u.shape[0])
# 75653 109975 8690 10191
# Percentage of positives in each training set
print(y_train.mean(), y_cps_o.mean(), y_cps_r.mean(), y_u.mean())
# 0.0668182358928 0.358054103205 0.436939010357 0.49602590521

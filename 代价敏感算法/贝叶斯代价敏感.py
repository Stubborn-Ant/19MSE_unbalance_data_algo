from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.models import BayesMinimumRiskClassifier
from costcla.metrics import savings_score
data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
f = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_prob_test = f.predict_proba(X_test)
y_pred_test_rf = f.predict(X_test)
f_bmr = BayesMinimumRiskClassifier()
f_bmr.fit(y_test, y_prob_test)
y_pred_test_bmr = f_bmr.predict(y_prob_test, cost_mat_test)
# Savings using only RandomForest
print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
# 0.12454256594
# Savings using RandomForest and Bayes Minimum Risk
print(savings_score(y_test, y_pred_test_bmr, cost_mat_test))
# 0.413425845555
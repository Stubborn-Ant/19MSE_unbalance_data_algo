from costcla.probcal import ROCConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from costcla.datasets import load_creditscoring1
from costcla.metrics import brier_score_loss

data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
f = RandomForestClassifier()
f.fit(X_train, y_train)
y_prob_test = f.predict_proba(X_test)

f_cal = ROCConvexHull()
f_cal.fit(y_test, y_prob_test)
y_prob_test_cal = f_cal.predict_proba(y_prob_test)
# Brier score using only RandomForest
print(brier_score_loss(y_test, y_prob_test[:, 1]))  # 0.0577615264881

# Brier score using calibrated RandomForest
print(brier_score_loss(y_test, y_prob_test_cal))  # 0.0553677407894

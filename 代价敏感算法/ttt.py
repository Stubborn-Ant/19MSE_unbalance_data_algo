from costcla.datasets import load_creditscoring1
from costcla.sampling import smote
data = load_creditscoring1()
data_smote, target_smote = smote(data.data, data.target, per=0.7)
# Size of each training set
print(data.data.shape[0], data_smote.shape[0])
# 112915 204307
# Percentage of positives in each training set
print(data.target.mean(), target_smote.mean())
# 0.0674489660364 0.484604051746
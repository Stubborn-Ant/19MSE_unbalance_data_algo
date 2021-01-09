import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style({'font.sans-serif':['SimHei','Arial']})
a = np.asarray(
    [
        [1, 0.108336159, 0.45590947, -0.158433645],
        [0.108336159, 1, 0.23277721, -0.23454366],
        [0.45590947, 0.23277721, 1, -0.272811593],
        [-0.158433645, -0.23454366, -0.272811593, 1]
    ]
)

sns.heatmap(a, cmap='Reds', xticklabels=['money', 'bad_expr', 'well_expr', 'retention'],
            yticklabels=['money', 'bad_expr', 'well_expr', 'retention'])
plt.show()

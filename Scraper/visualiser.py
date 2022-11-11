import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_json("repo_data/repo_analysis.json")

bins = [-1, 2, 3, 4, 5, 10, 20, 50, np.inf]
labels = ['2', '3', '4', '5', '6-10', '11-20', "21-50", "more than 50"]
df = df.groupby(pd.cut(df['forkcount'], bins=bins, labels=labels)).size().reset_index(name='count')

df.plot(x='forkcount', y='count', kind='bar')
plt.show()

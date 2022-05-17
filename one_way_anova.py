import pandas as pd
import researchpy as rp
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import numpy as np

# read file
df = pd.read_csv('proportions_data.csv')

# median = orange line, mean = green triangle

# generate p-values
stats_four = f_oneway(df['LIME-RISE'], df['LIME-SHAP'], df['RISE-SHAP'])
stats_lr_ls = f_oneway(df['LIME-RISE'], df['LIME-SHAP'])
stats_lr_rs = f_oneway(df['LIME-RISE'], df['RISE-SHAP'])
stats_ls_rs = f_oneway(df['LIME-SHAP'], df['RISE-SHAP'])
print(stats_lr_ls)
print(stats_lr_rs)
print(stats_ls_rs)

# generate other statistical values - mean, max, min, standard deviation
print(np.mean(df['LIME-RISE']))
print(max(df['LIME-RISE']))
print(min(df['LIME-RISE']))
print(np.std(df['LIME-RISE']))

print(np.mean(df['LIME-SHAP']))
print(max(df['LIME-SHAP']))
print(min(df['LIME-SHAP']))
print(np.std(df['LIME-SHAP']))

print(np.mean(df['RISE-SHAP']))
print(max(df['RISE-SHAP']))
print(min(df['RISE-SHAP']))
print(np.std(df['RISE-SHAP']))

print(np.mean(df['ALL']))
print(max(df['ALL']))
print(min(df['ALL']))
print(np.std(df['ALL']))

# generate box plot
fig = plt.figure(figsize= (7, 7))
ax = fig.add_subplot(111)
ax.set_title("Box Plot of Pixel Importance Agreement")
ax.set
data = df['LIME-RISE'], df['LIME-SHAP'], df['RISE-SHAP'], df['ALL']
ax.boxplot(data, labels= ['LIME-RISE', 'LIME-SHAP', 'RISE-SHAP', 'ALL'], showmeans=True)
plt.xlabel("Compared Methods")
plt.ylabel("Proportion of Agreed Pixels")
plt.savefig("final_anova.png")
plt.show()

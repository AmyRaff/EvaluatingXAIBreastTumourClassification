import pandas as pd

# read file
mylines = []
with open('rbo_vals.txt', 'rt') as myfile:
    for line in myfile:
        mylines.append(line)

# RBO -----------------------------------------------------------------------------------------------------------------
# convert to list of values
values = []
for i in range(len(mylines)):
    line = mylines[i]
    if line.count("RBO") > 0:
        ind = line.index("RBO") + 5
        value = line[ind:len(line)-1]
        values.append(value)


rise_shap_9 = []
rise_shap_95 = []
rise_shap_99 = []
lime_shap_9 = []
lime_shap_95 = []
lime_shap_99 = []
rise_lime_9 = []
rise_lime_95 = []
rise_lime_99 = []

# assign each value to the comparison they are part of
for v in range(len(values)):
    value = values[v]
    mod_val = v % 9
    if mod_val == 0: rise_shap_9.append(value)
    elif mod_val == 1: rise_shap_95.append(value)
    elif mod_val == 2: rise_shap_99.append(value)
    elif mod_val == 3: lime_shap_9.append(value)
    elif mod_val == 4: lime_shap_95.append(value)
    elif mod_val == 5: lime_shap_99.append(value)
    elif mod_val == 6: rise_lime_9.append(value)
    elif mod_val == 7: rise_lime_95.append(value)
    elif mod_val == 8: rise_lime_99.append(value)

# generate dataframe
outs = pd.DataFrame()
outs['RISE-SHAP-0.9'] = rise_shap_9
outs['RISE-SHAP-0.95'] = rise_shap_95
outs['RISE-SHAP-0.99'] = rise_shap_99
outs['LIME-SHAP-0.9'] = lime_shap_9
outs['LIME-SHAP-0.95'] = lime_shap_95
outs['LIME-SHAP-0.99'] = lime_shap_99
outs['RISE-LIME-0.9'] = rise_lime_9
outs['RISE-LIME-0.95'] = rise_lime_95
outs['RISE-LIME-0.99'] = rise_lime_99

# generate CSV file
outs.to_csv('rbo_data.csv', index=None)

# Kendall's Tau --------------------------------------------------------------------------------------------------

correlations = []
pvals = []
# generate list of values
for i in range(len(mylines)):
    line = mylines[i]
    if line.count("KENDALL") > 0:
        corr_ind = line.index("correlation") + 12
        pval_ind_start = line.index("pvalue")
        corr_value = line[corr_ind:pval_ind_start - 2]
        correlations.append(corr_value)
        pval = line[pval_ind_start + 7:len(line) - 2]
        pvals.append(pval)


rise_shap_full_ken_c, rise_shap_full_ken_p = [], []
rise_shap_5000_ken_c, rise_shap_5000_ken_p = [], []
rise_shap_1000_ken_c, rise_shap_1000_ken_p = [], []
rise_shap_n_ken_c, rise_shap_n_ken_p = [], []

lime_shap_full_ken_c, lime_shap_full_ken_p = [], []
lime_shap_5000_ken_c, lime_shap_5000_ken_p = [], []
lime_shap_1000_ken_c, lime_shap_1000_ken_p = [], []
lime_shap_n_ken_c, lime_shap_n_ken_p = [], []

rise_lime_full_ken_c, rise_lime_full_ken_p = [], []
rise_lime_5000_ken_c, rise_lime_5000_ken_p = [], []
rise_lime_1000_ken_c, rise_lime_1000_ken_p = [], []
rise_lime_n_ken_c, rise_lime_n_ken_p = [], []

# assign values to comparisons they are part of
for v in range(len(correlations)):
    c = correlations[v]
    p = pvals[v]
    mod_val = v % 12
    if mod_val == 0:
        rise_shap_full_ken_c.append(c)
        rise_shap_full_ken_p.append(p)
    elif mod_val == 1:
        rise_shap_5000_ken_c.append(c)
        rise_shap_5000_ken_p.append(p)
    elif mod_val == 2:
        rise_shap_1000_ken_c.append(c)
        rise_shap_1000_ken_p.append(p)
    elif mod_val == 3:
        rise_shap_n_ken_c.append(c)
        rise_shap_n_ken_p.append(p)
    elif mod_val == 4:
        lime_shap_full_ken_c.append(c)
        lime_shap_full_ken_p.append(p)
    elif mod_val == 5:
        lime_shap_5000_ken_c.append(c)
        lime_shap_5000_ken_p.append(p)
    elif mod_val == 6:
        lime_shap_1000_ken_c.append(c)
        lime_shap_1000_ken_p.append(p)
    elif mod_val == 7:
        lime_shap_n_ken_c.append(c)
        lime_shap_n_ken_p.append(p)
    elif mod_val == 8:
        rise_lime_full_ken_c.append(c)
        rise_lime_full_ken_p.append(p)
    elif mod_val == 9:
        rise_lime_5000_ken_c.append(c)
        rise_lime_5000_ken_p.append(p)
    elif mod_val == 10:
        rise_lime_1000_ken_c.append(c)
        rise_lime_1000_ken_p.append(p)
    elif mod_val == 11:
        rise_lime_n_ken_c.append(c)
        rise_lime_n_ken_p.append(p)

# generate dataframe
kendall_outs = pd.DataFrame()
kendall_outs['RISE-SHAP-FULL-C'] = rise_shap_full_ken_c
kendall_outs['RISE-SHAP-5000-C'] = rise_shap_5000_ken_c
kendall_outs['RISE-SHAP-1000-C'] = rise_shap_1000_ken_c
kendall_outs['RISE-SHAP-N-C'] = rise_shap_n_ken_c
kendall_outs['LIME-SHAP-FULL-C'] = lime_shap_full_ken_c
kendall_outs['LIME-SHAP-5000-C'] = lime_shap_5000_ken_c
kendall_outs['LIME-SHAP-1000-C'] = lime_shap_1000_ken_c
kendall_outs['LIME-SHAP-N-C'] = lime_shap_n_ken_c
kendall_outs['RISE-LIME-FULL-C'] = rise_lime_full_ken_c
kendall_outs['RISE-LIME-5000-C'] = rise_lime_5000_ken_c
kendall_outs['RISE-LIME-1000-C'] = rise_lime_1000_ken_c
kendall_outs['RISE-LIME-N-C'] = rise_lime_n_ken_c

kendall_outs['RISE-SHAP-FULL-P'] = rise_shap_full_ken_p
kendall_outs['RISE-SHAP-5000-P'] = rise_shap_5000_ken_p
kendall_outs['RISE-SHAP-1000-P'] = rise_shap_1000_ken_p
kendall_outs['RISE-SHAP-N-P'] = rise_shap_n_ken_p
kendall_outs['LIME-SHAP-FULL-P'] = lime_shap_full_ken_p
kendall_outs['LIME-SHAP-5000-P'] = lime_shap_5000_ken_p
kendall_outs['LIME-SHAP-1000-P'] = lime_shap_1000_ken_p
kendall_outs['LIME-SHAP-N-P'] = lime_shap_n_ken_p
kendall_outs['RISE-LIME-FULL-P'] = rise_lime_full_ken_p
kendall_outs['RISE-LIME-5000-P'] = rise_lime_5000_ken_p
kendall_outs['RISE-LIME-1000-P'] = rise_lime_1000_ken_p
kendall_outs['RISE-LIME-N-P'] = rise_lime_n_ken_p

# generate CSV file
kendall_outs.to_csv('kendall_data.csv', index=None)


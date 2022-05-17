import pandas as pd

mylines = []
# read the file
with open('out.txt', 'rt') as myfile:
    for line in myfile:              
        mylines.append(line)    

lime_shap = []
lime_rise = []
rise_shap = []
alls = []
values = []

# convert to list of numerical values
for i in range(len(mylines)):
    line = mylines[i]
    bracket_start = line.find("(") # -1 if no brackets present
    # value starts at index bracket_start + 1
    bracket_end = line.find(")")
    if bracket_start > -1:
        value = line[bracket_start + 1:bracket_end - 1]
        values.append(value)

# separate values into lists regarding which comparison they are for
for v in range(len(values)):
    value = values[v]
    mod_val = v % 4
    if mod_val == 0: lime_shap.append(value)
    elif mod_val == 1: lime_rise.append(value)
    elif mod_val == 2: rise_shap.append(value)
    else: alls.append(value)

# generate dataframe
outs = pd.DataFrame()
outs['LIME-SHAP'] = lime_shap
outs['LIME-RISE'] = lime_rise
outs['RISE-SHAP'] = rise_shap
outs['ALL'] = alls

# generate CSV file
outs.to_csv('proportions_data.csv', index=None)

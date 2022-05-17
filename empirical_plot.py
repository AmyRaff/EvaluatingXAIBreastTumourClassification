import matplotlib.pyplot as plt

# averages taken from excel file
three = [21.40078409, 17.67478165, 17.94017218, 5.509680766]
four = [22.82365657, 19.5124118, 20.85477279, 7.225894101]
five = [23.34453199, 22.47126095, 22.66367464, 8.080000244]
six = [26.87199663, 27.77035116, 23.94899888, 10.71931764]
seven = [27.21999395, 28.7119077, 25.12271987, 10.48355006]

titles = ["LIME-SHAP", "LIME-RISE", "RISE-SHAP", "ALL"]

num_features = [3, 4, 5, 6, 7]
lime_shap = [three[0], four[0], five[0], six[0], seven[0]]
lime_rise = [three[1], four[1], five[1], six[1], seven[1]]
rise_shap = [three[2], four[2], five[2], six[2], seven[2]]
alls = [three[3], four[3], five[3], six[3], seven[3]]

# plot figure
plt.plot(num_features, lime_shap, label=titles[0])
plt.plot(num_features, lime_rise, label=titles[1])
plt.plot(num_features, rise_shap, label=titles[2])
plt.plot(num_features, alls, label=titles[3])
plt.xticks(num_features)
plt.legend()
plt.xlabel("Number of LIME Features Used")
plt.ylabel("Average Percentage Agreement")
plt.title("Average % Pixel Agreement over Validation Set for different L Values")
plt.savefig("empirical_comparison.png")
plt.show()

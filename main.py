import helper
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from SHAP_explainer import shapExplainer
from RISE_explainer import riseExplainer
from LIME_explainer import limeExplainer
from scipy import stats

# This file runs the whole experiment - generates explanations (assuming image saving is toggled on in each file
# for LIME, RISE and SHAP), pixel agreement statistics, RBO values and Kendall's Tau values

IMG_WIDTH, IMG_HEIGHT = (227, 227)

# toggle validation data for experiments, test data for final results
# val_X, val_y = helper.X_val, helper.y_val
test_X, test_y = helper.X_test, helper.y_test

used_features = 6  # empirically chose 6

for ind in range(len(test_X) - 1):
    index = ind
    # pixel importance values from each explainer
    # NOTE: this saves explanations for each technique if the save image code is toggled on in the individual files
    shap_data = shapExplainer(index, test_X, test_y)
    rise_data = riseExplainer(index, test_X, test_y)
    lime_data, num_features = limeExplainer(index, test_X, test_y, used_features)

    # get pixels in top 6 features for LIME
    def get_lime_features():
        data = [k for k, v in lime_data.items() if float(v) == 1]
        xs = []
        ys = []
        for i in data:
            ys.append(227-i[0])
            xs.append(i[1])
        return xs, ys

    # function for getting the most important num_pixels pixels from a list
    def get_feature_plot_data(data, num_pixels):
        k = Counter(data)
        high = k.most_common(num_pixels)
        ks = []
        vs = []
        for i in high:
            ks.append(i[0])
            vs.append(i[1])
        xs = []
        ys = []
        for p in range(len(ks)):
            ys.append(227-ks[p][0])
            xs.append(ks[p][1])
        return xs, ys


    img = test_X[index]
    lime_xs, lime_ys = get_lime_features()
    num_pixels = len(lime_xs)  # number of pixels in 6 most important features
    print(num_pixels)
    # generate pixel lists for RISE and SHAP containing n most important pixels (equal length to LIME)
    shap_xs, shap_ys = get_feature_plot_data(shap_data, num_pixels)
    rise_xs, rise_ys = get_feature_plot_data(rise_data, num_pixels)
    # ----------------------------------------------------------------------------------------------------------------
    # Generate pixel agreement statistics

    l_s_agree = 0
    l_r_agree = 0
    r_s_agree = 0
    all_agree = 0

    for i in range(len(lime_xs)):
        for j in range(len(shap_xs)):
            if shap_xs[j] == lime_xs[i] and shap_ys[j] == lime_ys[i]:
                l_s_agree += 1
            if rise_xs[j] == lime_xs[i] and rise_ys[j] == lime_ys[i]:
                l_r_agree += 1

    l_s_percentage = 100 * l_s_agree / num_pixels
    l_r_percentage = 100 * l_r_agree / num_pixels

    print("Pixels where LIME and SHAP agree: {} ({}%)".format(l_s_agree, l_s_percentage))
    print("Pixels where LIME and RISE agree: {} ({}%)".format(l_r_agree, l_r_percentage))

    for i in range(len(shap_xs)):
        for j in range(len(rise_xs)):
            if shap_xs[i] == rise_xs[j] and shap_ys[i] == rise_ys[j]:
                r_s_agree += 1

    r_s_percentage = 100 * r_s_agree / num_pixels
    print("Pixels where RISE and SHAP agree: {} ({}%)".format(r_s_agree, r_s_percentage))

    for i in range(len(lime_xs)):
        for j in range(len(rise_xs)):
            if lime_xs[i] == rise_xs[j] and lime_ys[i] == rise_ys[j]:
                for k in range(len(shap_xs)):
                    if rise_xs[j] == shap_xs[k] and rise_ys[j] == shap_ys[k]:
                        all_agree += 1

    all_percentage = 100 * all_agree / num_pixels
    print("Pixels where all 3 methods agree: {} ({}%)".format(all_agree, all_percentage))

    # write to file for analysis
    with open('out.txt', 'a') as f:
        f.write("Image: {}\n".format(index))
        f.write("Pixels where LIME and SHAP agree: {} ({}%)\n".format(l_s_agree, l_s_percentage))
        f.write("Pixels where LIME and RISE agree: {} ({}%)\n".format(l_r_agree, l_r_percentage))
        f.write("Pixels where RISE and SHAP agree: {} ({}%)\n".format(r_s_agree, r_s_percentage))
        f.write("Pixels where all 3 methods agree: {} ({}%)\n".format(all_agree, all_percentage))

    # generate overlay plots to visualise agreement of top n pixels between techniques
    # toggle the save line and the show line as needed
    plt.imshow(img, zorder=0, extent=[0, 227, 0, 227])
    plt.scatter(rise_xs, rise_ys, zorder=1, s=2, alpha=0.1, c='blue')
    plt.scatter(shap_xs, shap_ys, zorder=1, s=2, alpha=0.1, c='red')
    plt.scatter(lime_xs, lime_ys, zorder=1, s=2, alpha=0.15, c='green')
    plt.xticks(np.arange(0, 227, 30))
    plt.legend(['SHAP', 'RISE', 'LIME'], prop={'size': 12})
    plt.title("Top n Most Important Pixels")
    plt.yticks(np.arange(0, 227, 30))
    plt.savefig("all-{}-plot.png".format(index))
    # plt.show()

    # --------------------------------------------------------------------------------------------------------------
    # RBO results

    # This function is taken from the public Python implementation https://github.com/ragrawal/measures
    def rbo(l1, l2, p):
        """
            Calculates Ranked Biased Overlap (RBO) score.
            l1 -- Ranked List 1
            l2 -- Ranked List 2
        """
        if l1 == None: l1 = []
        if l2 == None: l2 = []
        sl,ll = sorted([(len(l1), l1),(len(l2),l2)])
        s, S = sl
        l, L = ll
        if s == 0: return 0
        # Calculate the overlaps at ranks 1 through l (the longer of the two lists)
        ss = set([])  # contains elements from the smaller list till depth i
        ls = set([])  # contains elements from the longer list till depth i
        x_d = {0: 0}
        sum1 = 0.0
        for i in range(l):
            x = L[i]
            y = S[i] if i < s else None
            d = i + 1
            # if two elements are same then we don't need to add to either of the set
            if x == y:
                x_d[d] = x_d[d-1] + 1.0
            # else add items to respective list and calculate overlap
            else:
                ls.add(x)
                if y!=None: ss.add(y)
                x_d[d] = x_d[d-1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
                # calculate average overlap
            sum1 += x_d[d]/d * pow(p, d)
        sum2 = 0.0
        for i in range(l-s):
            d = s+i+1
            sum2 += x_d[d]*(d-s)/(d*s)*pow(p,d)
        sum3 = ((x_d[l]-x_d[s])/l+x_d[s]/s)*pow(p,l)
        rbo_ext = (1-p)/p*(sum1+sum2)+sum3
        return rbo_ext

    # sort the pixel lists, and generate sub-lists for top n, 5000 and 1000 pixels for experiments
    r_marklist = sorted(rise_data.items(), key=lambda x:x[1], reverse=True)
    r_sortdict = dict(r_marklist)  # descending
    rise_pixels = []
    for key in r_sortdict.keys():
        rise_pixels.append("{}".format(key))
    rise_pixels_1000 = rise_pixels[:1000]
    rise_pixels_5000 = rise_pixels[:5000]
    rise_pixels_n = rise_pixels[:num_pixels]

    # sort the pixel lists, and generate sub-lists for top n, 5000 and 1000 pixels for experiments
    s_marklist = sorted(shap_data.items(), key=lambda x:x[1], reverse=True)
    s_sortdict = dict(s_marklist)  # descending
    shap_pixels = []
    for key in s_sortdict.keys():
        shap_pixels.append("{}".format(key))
    shap_pixels_1000 = shap_pixels[:1000]
    shap_pixels_5000 = shap_pixels[:5000]
    shap_pixels_n = shap_pixels[:num_pixels]

    # sort the pixel lists, and generate sub-lists for top n, 5000 and 1000 pixels for experiments
    l_marklist = sorted(lime_data.items(), key=lambda x:x[1], reverse=True)
    l_sortdict = dict(l_marklist)  # descending
    lime_pixels = []
    for key in l_sortdict.keys():
        lime_pixels.append("{}".format(key))
    lime_pixels_1000 = lime_pixels[:1000]
    lime_pixels_5000 = lime_pixels[:5000]
    lime_pixels_n = lime_pixels[:num_pixels]

    # generate RBO values
    r_s_rbo_9 = rbo(rise_pixels, shap_pixels, 0.9)
    r_s_rbo_95 = rbo(rise_pixels, shap_pixels, 0.95)
    r_s_rbo_99 = rbo(rise_pixels, shap_pixels, 0.99)
    print("RISE-SHAP p=0.9 RBO: {}".format(r_s_rbo_9))
    print("RISE-SHAP p=0.95 RBO: {}".format(r_s_rbo_95))
    print("RISE-SHAP p=0.99 RBO: {}".format(r_s_rbo_99))
    l_s_rbo_9 = rbo(lime_pixels, shap_pixels, 0.9)
    l_s_rbo_95 = rbo(lime_pixels, shap_pixels, 0.95)
    l_s_rbo_99 = rbo(lime_pixels, shap_pixels, 0.99)
    print("LIME-SHAP p=0.9 RBO: {}".format(l_s_rbo_9))
    print("LIME-SHAP p=0.95 RBO: {}".format(l_s_rbo_95))
    print("LIME-SHAP p=0.99 RBO: {}".format(l_s_rbo_99))
    r_l_rbo_9 = rbo(rise_pixels, lime_pixels, 0.9)
    r_l_rbo_95 = rbo(rise_pixels, lime_pixels, 0.95)
    r_l_rbo_99 = rbo(rise_pixels, lime_pixels, 0.99)
    print("RISE-LIME p=0.9 RBO: {}".format(r_l_rbo_9))
    print("RISE-LIME p=0.95 RBO: {}".format(r_l_rbo_95))
    print("RISE-LIME p=0.99 RBO: {}".format(r_l_rbo_99))

    # write values to file for experiments
    with open('rbo_vals.txt', 'a') as f:
        f.write("Image: {}\n".format(index))
        f.write("RISE-SHAP p=0.9 RBO: {}\n".format(r_s_rbo_9))
        f.write("RISE-SHAP p=0.95 RBO: {}\n".format(r_s_rbo_95))
        f.write("RISE-SHAP p=0.99 RBO: {}\n".format(r_s_rbo_99))
        f.write("LIME-SHAP p=0.9 RBO: {}\n".format(l_s_rbo_9))
        f.write("LIME-SHAP p=0.95 RBO: {}\n".format(l_s_rbo_95))
        f.write("LIME-SHAP p=0.99 RBO: {}\n".format(l_s_rbo_99))
        f.write("RISE-LIME p=0.9 RBO: {}\n".format(r_l_rbo_9))
        f.write("RISE-LIME p=0.95 RBO: {}\n".format(r_l_rbo_95))
        f.write("RISE-LIME p=0.99 RBO: {}\n".format(r_l_rbo_99))

    # ----------------------------------------------------------------------------------------------------------------
    # Kendall's Tau results

    # generate values
    r_s_kendall = stats.kendalltau(rise_pixels, shap_pixels)
    r_s_kendall_1000 = stats.kendalltau(rise_pixels_1000, shap_pixels_1000)
    r_s_kendall_5000 = stats.kendalltau(rise_pixels_5000, shap_pixels_5000)
    r_s_kendall_n = stats.kendalltau(rise_pixels_n, shap_pixels_n)
    l_s_kendall = stats.kendalltau(lime_pixels, shap_pixels)
    l_s_kendall_1000 = stats.kendalltau(lime_pixels_1000, shap_pixels_1000)
    l_s_kendall_5000 = stats.kendalltau(lime_pixels_5000, shap_pixels_5000)
    l_s_kendall_n = stats.kendalltau(lime_pixels_n, shap_pixels_n)
    r_l_kendall = stats.kendalltau(rise_pixels, lime_pixels)
    r_l_kendall_1000 = stats.kendalltau(rise_pixels_1000, lime_pixels_1000)
    r_l_kendall_5000 = stats.kendalltau(rise_pixels_5000, lime_pixels_5000)
    r_l_kendall_n = stats.kendalltau(rise_pixels_n, lime_pixels_n)

    # write values to file
    with open('rbo_vals.txt', 'a') as f:
        f.write("Image: {}\n".format(index))
        f.write("RISE-SHAP FULL KENDALL: {}\n".format(r_s_kendall))
        f.write("RISE-SHAP TOP 5000 KENDALL: {}\n".format(r_s_kendall_5000))
        f.write("RISE-SHAP TOP 1000 KENDALL: {}\n".format(r_s_kendall_1000))
        f.write("RISE-SHAP N KENDALL: {}\n".format(r_s_kendall_n))
        f.write("LIME-SHAP FULL KENDALL: {}\n".format(l_s_kendall))
        f.write("LIME-SHAP TOP 5000 KENDALL: {}\n".format(l_s_kendall_5000))
        f.write("LIME-SHAP TOP 1000 KENDALL: {}\n".format(l_s_kendall_1000))
        f.write("LIME-SHAP N KENDALL: {}\n".format(l_s_kendall_n))
        f.write("RISE-LIME FULL KENDALL: {}\n".format(r_l_kendall))
        f.write("RISE-LIME TOP 5000 KENDALL: {}\n".format(r_l_kendall_5000))
        f.write("RISE-LIME TOP 1000 KENDALL: {}\n".format(r_l_kendall_1000))
        f.write("RISE-LIME N KENDALL: {}\n".format(r_l_kendall_n))


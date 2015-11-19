from __future__ import print_function, division
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt


import pickle



def SVM():
    return True
def Naive_Bayes():
    return True
def Decision_Tree():
    return True


if __name__ == '__main__':

    methods=[SVM,Naive_Bayes,Decision_Tree]
    #issmote=["smote","no_smote"]





    with open('../dump/result_means.pickle', 'rb') as handle:
        Y_median=pickle.load(handle)
        Y_iqr=pickle.load(handle)
        trace_feature_num=pickle.load(handle)

###############################################################

    issmote=["no_smote"]

    "draw"
    plt.figure(num=0,figsize=(16,6))
    plt.subplot(121)
    for is_smote in issmote:
        for method in methods:

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean"],label="median_"+method.__name__)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean"],"-.",color=line.get_color(),label="iqr_"+method.__name__)


    plt.ylabel("Fscore_M")
    plt.xlabel("Number of Features")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("../fig/result_imbalance_unweighted.png")


    "draw"
    plt.figure(num=1,figsize=(16,6))
    plt.subplot(121)
    for is_smote in issmote:
        for method in methods:

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean_weighted"],label="median_"+method.__name__)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean_weighted"],"-.",color=line.get_color(),label="iqr_"+method.__name__)

    plt.ylabel("Fscore_mu")
    plt.xlabel("Number of Features")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("../fig/result_imbalance_weighted.png")



    keys=Y_median["no_smote"][SVM.__name__].keys()
    keys.remove("mean_weighted")
    keys.remove("mean")

    count={'others': 1609, 'identification-request': 1099, 'naruto': 547, 'one-piece': 204, 'anime-production': 164, 'fullmetal-alchemist': 114, 'tropes': 110, 'bleach': 87, 'death-note': 86, 'fairy-tail': 78, 'dragon-ball': 66, 'code-geass': 51, 'japanese-language': 49, 'sword-art-online': 48, 'monogatari-series': 46, 'madoka-magica': 46, 'culture': 45, 'pokemon': 45, 'fate-stay-night': 43, 'shingeki-no-kyojin': 40, 'hunter-x-hunter': 38, 'anime-history': 35, 'manga-production': 33, 'from-the-new-world': 26, 'terminology': 25, 'neon-genesis-evangelion': 24, 'music': 24, 'toaru-majutsu-no-index': 23, 'resources': 22}
    keys=np.array(count.keys())
    pop=count.values()
    ind=np.argsort(pop)
    keys=keys[ind]

    x=range(len(keys))

    "draw"
    plt.figure(num=2,figsize=(32,6))
    plt.subplot(121)
    for is_smote in issmote:
        for method in methods:
            med_Y=[]
            iqr_Y=[]
            for key in keys:
                med_Y.append(Y_median[is_smote][method.__name__][key][2])
                iqr_Y.append(Y_iqr[is_smote][method.__name__][key][2])
            line,=plt.plot(x,med_Y,label="median_"+method.__name__)
            plt.plot(x,iqr_Y,"-.",color=line.get_color(),label="iqr_"+method.__name__)

    keys_abr=[]
    for key in keys:
        keys_abr.append(key[:3])
    plt.xticks(x, keys_abr)
    plt.ylabel("F score")
    plt.xlabel("Classes")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("../fig/result_imbalance_classes.png")

###############################################################

    issmote=["smote"]

    "draw"
    plt.figure(num=3,figsize=(16,6))
    plt.subplot(121)
    for is_smote in issmote:
        for method in methods:

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean"],label="median_"+method.__name__)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean"],"-.",color=line.get_color(),label="iqr_"+method.__name__)


    plt.ylabel("Fscore_M")
    plt.xlabel("Number of Features")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("../fig/result_balance_unweighted.png")


    "draw"
    plt.figure(num=4,figsize=(16,6))
    plt.subplot(121)
    for is_smote in issmote:
        for method in methods:

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean_weighted"],label="median_"+method.__name__)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean_weighted"],"-.",color=line.get_color(),label="iqr_"+method.__name__)

    plt.ylabel("Fscore_mu")
    plt.xlabel("Number of Features")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("../fig/result_balance_weighted.png")



    keys=Y_median["no_smote"][SVM.__name__].keys()
    keys.remove("mean_weighted")
    keys.remove("mean")

    count={'others': 1609, 'identification-request': 1099, 'naruto': 547, 'one-piece': 204, 'anime-production': 164, 'fullmetal-alchemist': 114, 'tropes': 110, 'bleach': 87, 'death-note': 86, 'fairy-tail': 78, 'dragon-ball': 66, 'code-geass': 51, 'japanese-language': 49, 'sword-art-online': 48, 'monogatari-series': 46, 'madoka-magica': 46, 'culture': 45, 'pokemon': 45, 'fate-stay-night': 43, 'shingeki-no-kyojin': 40, 'hunter-x-hunter': 38, 'anime-history': 35, 'manga-production': 33, 'from-the-new-world': 26, 'terminology': 25, 'neon-genesis-evangelion': 24, 'music': 24, 'toaru-majutsu-no-index': 23, 'resources': 22}
    keys=np.array(count.keys())
    pop=count.values()
    ind=np.argsort(pop)
    keys=keys[ind]

    x=range(len(keys))

    "draw"
    plt.figure(num=5,figsize=(32,6))
    plt.subplot(121)
    for is_smote in issmote:
        for method in methods:
            med_Y=[]
            iqr_Y=[]
            for key in keys:
                med_Y.append(Y_median[is_smote][method.__name__][key][2])
                iqr_Y.append(Y_iqr[is_smote][method.__name__][key][2])
            line,=plt.plot(x,med_Y,label="median_"+method.__name__)
            plt.plot(x,iqr_Y,"-.",color=line.get_color(),label="iqr_"+method.__name__)

    keys_abr=[]
    for key in keys:
        keys_abr.append(key[:3])
    plt.xticks(x, keys_abr)
    plt.ylabel("F score")
    plt.xlabel("Classes")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("../fig/result_balance_classes.png")

###############################################################

    issmote=["no_smote","smote"]
    for i,method in enumerate(methods):
        fig_num=6+i*3

        "draw"
        plt.figure(num=fig_num,figsize=(16,6))
        plt.subplot(121)
        for is_smote in issmote:

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean"],label="median_"+is_smote)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean"],"-.",color=line.get_color(),label="iqr_"+is_smote)


        plt.ylabel("Fscore_M")
        plt.xlabel("Number of Features")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        plt.savefig("../fig/result_"+method.__name__+"_unweighted.png")


        "draw"
        plt.figure(num=fig_num+1,figsize=(16,6))
        plt.subplot(121)
        for is_smote in issmote:

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean_weighted"],label="median_"+is_smote)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean_weighted"],"-.",color=line.get_color(),label="iqr_"+is_smote)

        plt.ylabel("Fscore_mu")
        plt.xlabel("Number of Features")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        plt.savefig("../fig/result_"+method.__name__+"_weighted.png")



        keys=Y_median["no_smote"][SVM.__name__].keys()
        keys.remove("mean_weighted")
        keys.remove("mean")

        count={'others': 1609, 'identification-request': 1099, 'naruto': 547, 'one-piece': 204, 'anime-production': 164, 'fullmetal-alchemist': 114, 'tropes': 110, 'bleach': 87, 'death-note': 86, 'fairy-tail': 78, 'dragon-ball': 66, 'code-geass': 51, 'japanese-language': 49, 'sword-art-online': 48, 'monogatari-series': 46, 'madoka-magica': 46, 'culture': 45, 'pokemon': 45, 'fate-stay-night': 43, 'shingeki-no-kyojin': 40, 'hunter-x-hunter': 38, 'anime-history': 35, 'manga-production': 33, 'from-the-new-world': 26, 'terminology': 25, 'neon-genesis-evangelion': 24, 'music': 24, 'toaru-majutsu-no-index': 23, 'resources': 22}
        keys=np.array(count.keys())
        pop=count.values()
        ind=np.argsort(pop)
        keys=keys[ind]

        x=range(len(keys))

        "draw"
        plt.figure(num=fig_num+2,figsize=(32,6))
        plt.subplot(121)
        for is_smote in issmote:
            med_Y=[]
            iqr_Y=[]
            for key in keys:
                med_Y.append(Y_median[is_smote][method.__name__][key][2])
                iqr_Y.append(Y_iqr[is_smote][method.__name__][key][2])
            line,=plt.plot(x,med_Y,label="median_"+is_smote)
            plt.plot(x,iqr_Y,"-.",color=line.get_color(),label="iqr_"+is_smote)

        keys_abr=[]
        for key in keys:
            keys_abr.append(key[:3])
        plt.xticks(x, keys_abr)
        plt.ylabel("F score")
        plt.xlabel("Classes")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
        plt.savefig("../fig/result_"+method.__name__+"_classes.png")

###############################################################

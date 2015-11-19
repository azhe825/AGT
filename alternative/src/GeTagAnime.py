from __future__ import print_function, division
from collections import Counter
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
import sklearn.naive_bayes

from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import FeatureHasher
from random import randint,random,seed,shuffle
from time import time
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import *

"The following two are from RAISE Lab library"
from ABCD import ABCD
from sk import rdivDemo




"Decorator to report arguments and time taken"
def run(func):
    def inner(*args, **kwargs):
        t0=time()
        print("You are running: %s" % func.__name__)
        print("Arguments were: %s, %s"%(args, kwargs))
        result = func(*args, **kwargs)
        print("Time taken: %f secs"%(time()-t0))
        return result
    return inner

def timer(func):
    def inner(*args, **kwargs):
        t0=time()
        result= func(*args,**kwargs)
        print("%s takes time: %s secs" %(func.__name__,time()-t0))
        return result
    return inner


"vocabulary"
def vocabulary(lst_of_words):
  v = []
  for c in lst_of_words:
    v.extend(c[1:])
  return list(set(v))

"term frequency "
def token_freqs(doc):
    return Counter(doc[1:])


"tf"
def tf(corpus):
    mat=[token_freqs(doc) for doc in corpus]
    return mat

"tf-idf"
def tf_idf(corpus):
    word={}
    doc={}
    docs=0
    mat=[]
    for row_c in corpus:
        mat,word,doc,docs=tf_idf_inc(row_c,word,doc,docs,mat)
    tfidf={}
    words=sum(word.values())
    for key in doc.keys():
        tfidf[key]=word[key]/words*np.log(docs/doc[key])
    return mat,tfidf

"tf-idf_incremental"
def tf_idf_inc(row_c,word,doc,docs,mat):
    docs+=1
    row=token_freqs(row_c)
    mat.append(row)
    for key in row.keys():
        try:
            word[key]+=row[key]
        except:
            word[key]=row[key]
        try:
            doc[key]+=1
        except:
            doc[key]=1

    return mat,word,doc,docs


"L2 normalization"
def l2normalize(mat):
    mat=mat.astype(float)
    for i,row in enumerate(mat):
        nor=np.linalg.norm(row,2)
        if not nor==0:
            mat[i]=row/nor
    return mat

"hashing trick"
def hash(mat,n_features=100):
    hasher = FeatureHasher(n_features=n_features)
    X = hasher.transform(mat)
    X=X.toarray()
    return X

"make feature matrix"
def make_feature(corpus,sel="tfidf",norm=l2normalize,n_features=4000):
    label=list(zip(*corpus)[0])
    if sel=="tfidf":
        mat,tfidf=tf_idf(corpus)
        keys=np.array(tfidf.keys())[np.argsort(tfidf.values())][-n_features:]
        matt=[]
        for row in mat:
            matt.append([row[key] for key in keys])
        matt=np.array(matt)
        matt=norm(matt)

        '''
        "Store tfidf_temp for drawing"
        global tfidf_temp
        if filename_global not in tfidf_temp.keys():
            tfidf_temp[filename_global]=np.sort(tfidf.values())
        '''


    else:
        mat=tf(corpus)
        matt=hash(mat,n_features=n_features)
        matt=norm(matt)
    return matt,label



"""
"sample without replacement"
def sample_norep(data,num,k=0):
    corpus=[]
    for i in range(0,num):
        try: corpus.append(data.pop(randint(0,len(data)-1)))
        except: pass
    return corpus
"""


"smote"
def smote(data,label,num,k=5):
    labellist=list(set(label))
    dict={}
    for l in labellist:
        corpus=[]
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(sub)
        distances, indices = nbrs.kneighbors(sub)
        for i in range(0,num):
            mid=randint(0,len(sub)-1)
            nn=indices[mid,randint(1,k)]
            datamade=[]
            for j in range(0,len(sub[mid])):
                gap=random()
                datamade.append((sub[nn,j]-sub[mid,j])*gap+sub[mid,j])
            corpus.append(datamade)
        dict[l]=np.array(corpus)
    return dict

"smote to meet the number of majority"
def smote_max(data,label,k=5):
    labelCont=Counter(label)
    num=np.max(labelCont.values())
    labelmade=[]
    datamade=[]
    for l in labelCont:
        id=[i for i,x in enumerate(label) if x==l]
        sub=data[id]
        labelmade+=[l]*num
        if len(sub)==num:
            datamade.extend(sub)
            continue
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(sub)
        distances, indices = nbrs.kneighbors(sub)
        for i in range(0,num):
            mid=randint(0,len(sub)-1)
            nn=indices[mid,randint(1,k)]
            newp=[]
            for j in range(0,len(sub[mid])):
                gap=random()
                newp.append((sub[nn,j]-sub[mid,j])*gap+sub[mid,j])
            datamade.append(newp)

    datamade=np.array(datamade)
    labelmade=np.array(labelmade)
    return datamade, labelmade







"cross validation"
def cross_validation(data,label,methods=[],fold=5,is_smote="smote",k=5):

    def sample_cross(data,label,fold,index):
        l=int(len(data)/fold)
        ind_test=range(index*l,(index+1)*l)
        ind_train=range(index*l)+range((index+1)*l,len(data))
        data_train=data[ind_train]
        data_test=data[ind_test]
        label_train=np.array(label)[ind_train]
        label_test=np.array(label)[ind_test]
        return {'train': data_train,'test': data_test, 'label_train': label_train, 'label_test': label_test}

    F={}
    for method in methods:
        F[method.__name__]={}

    for i in range(fold):
        ind=range(len(label))
        shuffle(ind)
        label=np.array(label)[ind]
        data=data[ind]
        for index in range(fold):
            result=sample_cross(data,label,fold,index)
            data_train=result["train"]
            data_test=result["test"]
            label_train=result["label_train"]
            label_test=result["label_test"]
            if is_smote=="smote":
                data_train,label_train=smote_max(data_train,label_train,k)


            for method in methods:
                F[method.__name__]=method(data_train,data_test,label_train,label_test,F[method.__name__])

    return F


"SVM"
def SVM(train_data,test_data,train_label,test_label,F):
    clf = svm.LinearSVC(dual=False)
    clf.fit(train_data, train_label)
    prediction=clf.predict(test_data)
    abcd=ABCD(before=test_label,after=prediction)
    ll=list(set(test_label))
    tmp = np.array([k.stats()[-2] for k in abcd()])
    for i,v in enumerate(tmp):
        try:
            F[ll[i]].append(v)
        except:
            F[ll[i]]=[v]

    tC = Counter(test_label)
    FreqClass=[tC[kk]/len(test_label) for kk in ll]
    try:
        F["mean"].append(np.mean(tmp))
    except:
        F["mean"]=[np.mean(tmp)]
    try:
        F["mean_weighted"].append(np.sum(tmp*FreqClass))
    except:
        F["mean_weighted"]=[np.sum(tmp*FreqClass)]

    return F

"Decision Tree"
def Decision_Tree(train_data,test_data,train_label,test_label,F):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_label)
    prediction=clf.predict(test_data)
    abcd=ABCD(before=test_label,after=prediction)
    ll=list(set(test_label))
    tmp = np.array([k.stats()[-2] for k in abcd()])
    for i,v in enumerate(tmp):
        try:
            F[ll[i]].append(v)
        except:
            F[ll[i]]=[v]

    tC = Counter(test_label)
    FreqClass=[tC[kk]/len(test_label) for kk in ll]
    try:
        F["mean"].append(np.mean(tmp))
    except:
        F["mean"]=[np.mean(tmp)]
    try:
        F["mean_weighted"].append(np.sum(tmp*FreqClass))
    except:
        F["mean_weighted"]=[np.sum(tmp*FreqClass)]

    return F


"Naive Bayes"
def Naive_Bayes(train_data,test_data,train_label,test_label,F):
    #clf = sklearn.naive_bayes.BernoulliNB()
    clf = sklearn.naive_bayes.MultinomialNB()
    #clf = sklearn.naive_bayes.GaussianNB()
    clf.fit(train_data, train_label)
    prediction=clf.predict(test_data)
    abcd=ABCD(before=test_label,after=prediction)
    ll=list(set(test_label))
    tmp = np.array([k.stats()[-2] for k in abcd()])
    for i,v in enumerate(tmp):
        try:
            F[ll[i]].append(v)
        except:
            F[ll[i]]=[v]

    tC = Counter(test_label)
    FreqClass=[tC[kk]/len(test_label) for kk in ll]
    try:
        F["mean"].append(np.mean(tmp))
    except:
        F["mean"]=[np.mean(tmp)]
    try:
        F["mean_weighted"].append(np.sum(tmp*FreqClass))
    except:
        F["mean_weighted"]=[np.sum(tmp*FreqClass)]

    return F


"Change the number of features"
def feature_num_change(filename='',filepath='',filetype='.txt',thres=20,is_smote="smote",methods=[],
                       neighbors=5,norm=l2normalize,fold=5,n_range=10):
    corpus=readfile(filename=filepath+filename+filetype,thres=thres)

    feature_num=1000
    trace_feature_num=[]
    F_final={}
    for i in range(1,n_range+1):
        n_feature=int(feature_num*i)
        data,label=make_feature(corpus,sel="tfidf",norm=norm,n_features=n_feature)

        F=cross_validation(data,label,methods=methods,fold=fold,is_smote=is_smote,k=neighbors)

        F_final[str(n_feature)]=F
        trace_feature_num.append(n_feature)

        print(is_smote+"_"+str(n_feature))

    return {"F_final": F_final,"trace_feature_num": trace_feature_num}



"Preprocessing: stemming + stopwords removing"
def process(txt):
  stemmer = PorterStemmer()
  cachedStopWords = stopwords.words("english")
  return ' '.join([stemmer.stem(word) for word \
                   in txt.lower().split() if word not \
                   in cachedStopWords and len(word)>1])


"Load data from file to list of lists"
def readfile(filename='',thres=20):
    corpus=[]
    targetlist=[]
    labellst=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            doc=doc.lower()
            try:
                doc=doc.lower()
                label=doc.split(' >>> ')[1].split()[0]
                labellst.append(label)
                corpus.append([label]+process(doc.split(' >>> ')[0]).split())
            except: pass
    labelcount=Counter(labellst)
    labellst=list(set(labellst))
    for label in labellst:
        if labelcount[label]>thres:
            targetlist.append(label)
    targetlist.append('others')
    for doc in corpus:
        if doc[0] not in targetlist:
            doc[0]='others'
    return corpus





if __name__ == '__main__':
    filename='anime'
    filepath='../data/'
    thres=20
    F_feature_num={}
    methods=[SVM,Naive_Bayes,Decision_Tree]
    issmote=["smote","no_smote"]
    #issmote=["no_smote"]

    for is_smote in issmote:
        result=feature_num_change(filename=filename,filepath=filepath,filetype='.txt',thres=thres,is_smote=is_smote,
                                  methods=methods,neighbors=5,norm=l2normalize,n_range=10)
        F_feature_num[is_smote]=result['F_final']
        trace_feature_num=result['trace_feature_num']

    with open('../dump/result.pickle', 'wb') as handle:
        pickle.dump(F_feature_num, handle)
        pickle.dump(trace_feature_num,handle)





    "draw"
    plt.figure(num=0,figsize=(16,6))
    plt.subplot(121)
    Y_median={}
    Y_iqr={}
    for is_smote in issmote:
        Y_median[is_smote]={}
        Y_iqr[is_smote]={}
        for method in methods:
            Y_median[is_smote][method.__name__]={}
            Y_iqr[is_smote][method.__name__]={}
            for f_num in trace_feature_num:
                for key in F_feature_num[is_smote][str(f_num)][method.__name__]:
                    try:
                        Y_median[is_smote][method.__name__][key].append(np.median(F_feature_num[is_smote][str(f_num)][method.__name__][key]))
                        Y_iqr[is_smote][method.__name__][key].append(np.percentile(F_feature_num[is_smote][str(f_num)][method.__name__][key],75)-np.percentile(F_feature_num[is_smote][str(f_num)][method.__name__][key],25))
                    except:
                        Y_median[is_smote][method.__name__][key]=[np.median(F_feature_num[is_smote][str(f_num)][method.__name__][key])]
                        Y_iqr[is_smote][method.__name__][key]=[np.percentile(F_feature_num[is_smote][str(f_num)][method.__name__][key],75)-np.percentile(F_feature_num[is_smote][str(f_num)][method.__name__][key],25)]

            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean"],label="median_unweighted_"+is_smote+"_"+method.__name__)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean"],"-.",color=line.get_color(),label="iqr_unweighted_"+is_smote+"_"+method.__name__)
            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean_weighted"],label="median_weighted_"+is_smote+"_"+method.__name__)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean_weighted"],"-.",color=line.get_color(),label="iqr_weighted_"+is_smote+"_"+method.__name__)

            """
            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean"],label="median_unweighted_"+method.__name__+"_"+is_smote)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean"],"-.",color=line.get_color(),label="iqr_unweighted_"+method.__name__+"_"+is_smote)
            line,=plt.plot(trace_feature_num,Y_median[is_smote][method.__name__]["mean_weighted"],label="median_weighted_"+method.__name__+"_"+is_smote)
            plt.plot(trace_feature_num,Y_iqr[is_smote][method.__name__]["mean_weighted"],"-.",color=line.get_color(),label="iqr_weighted_"+method.__name__+"_"+is_smote)
            """

    plt.ylabel("F score")
    plt.xlabel("Number of Features")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("result.png")


    with open('../dump/result_means.pickle', 'wb') as handle:
        pickle.dump(Y_median, handle)
        pickle.dump(Y_iqr, handle)
        pickle.dump(trace_feature_num,handle)


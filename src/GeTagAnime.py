from __future__ import print_function, division
from collections import Counter
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import FeatureHasher
from random import randint,random,seed,shuffle
from time import time

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
    mat=tf(corpus)
    l=len(corpus)
    flat=[]
    for row in mat:
        flat+=row.keys()
    n=Counter(flat)
    for row in mat:
        for token in row:
            row[token]=row[token]*l/n[token]
    return mat


"L2 normalization"
def l2normalize(mat):
    for row in mat:
        n=0
        for key in row:
            n+=row[key]**2
        n=n**0.5
        for key in row:
            row[key]=row[key]/n
    return mat

"hashing trick"
def hash(mat,n_features=100):
    hasher = FeatureHasher(n_features=n_features)
    X = hasher.transform(mat)
    X=X.toarray()
    return X

"make feature matrix"
def make_feature(corpus,method=tf,norm=l2normalize,n_features=100):
    label=list(zip(*corpus)[0])
    mat=method(corpus)
    mat=norm(mat)
    mat=hash(mat,n_features=n_features)
    return mat,label



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

"sample"
def sample(data,data_smote,label,num_train,num_test):

    if id(data)==id(data_smote):
        ind=np.random.choice(len(data),(num_train+num_test),replace=False)
        train_ind=ind[:num_train]
        test_ind=ind[num_train:]
        data_train=data[train_ind]
        data_test=data[test_ind]
        label_train=np.array(label)[train_ind]
        label_test=np.array(label)[test_ind]
    else:
        label_list=data_smote.keys()
        num=int(num_train/len(label_list))
        data_train=[]
        label_train=[]
        for l in label_list:
            if data_train==[]:
                data_train=data_smote[l][np.random.choice(len(data_smote[l]),num,replace=False)]
            else:
                data_train=np.vstack((data_train,data_smote[l][np.random.choice(len(data_smote[l]),num,replace=False)]))
            label_train.extend([l]*num)
        label_train=np.array(label_train)
        tmp=range(0,len(label_train))
        shuffle(tmp)
        data_train=data_train[tmp]
        label_train=label_train[tmp]
        test_ind=np.random.choice(len(data),num_test,replace=False)
        data_test=data[test_ind]
        label_test=np.array(label)[test_ind]


    return {'train': data_train,'test': data_test, 'label_train': label_train, 'label_test': label_test}




"sample_training"
def sample_training(data,data_smote,label,num_train,num_test,repeats=30):

    for i in range(0,repeats):

        result=sample(data,data_smote,label,num_train,num_test)


        data_train=result["train"]
        data_test=result["test"]
        label_train=result["label_train"]
        label_test=result["label_test"]


        "SVM"
        F=do_SVM(data_train,data_test,label_train,label_test)

        yield F


"SVM"
def do_SVM(train_data,test_data,train_label,test_label):
    clf = svm.LinearSVC(dual=False)
    clf.fit(train_data, train_label)
    prediction=clf.predict(test_data)
    abcd=ABCD(before=test_label,after=prediction)
    F = np.array([k.stats()[-2] for k in abcd()])
    tC = Counter(test_label)
    FreqClass=[tC[kk]/len(test_label) for kk in list(set(test_label))]
    ExptF = np.sum(F*FreqClass)
    return ExptF

"Change the number of features"
@run
def feature_num_change(filename='',filepath='',filetype='.txt',thres=20,issmote="smote",
                       neighbors=5,feature=tf,norm=l2normalize,repeats=30,n_range=10):
    corpus=readfile(filename=filepath+filename+filetype,thres=thres)
    #feature_num=len(vocabulary(corpus))

    feature_num=5120

    F_feature_num=[]
    trace_feature_num=[]
    for i in range(1,n_range+1):
        n_feature=int(feature_num*(0.5)**(n_range-i))
        data,label=make_feature(corpus,method=feature,norm=norm,n_features=n_feature)

        if issmote=="smote":
            num=int(len(data)/len(set(label)))
            data_smote=smote(data,label,num,k=neighbors)
        else: data_smote=data

        num_train=int(len(data)*0.9)
        num_test=int(len(data)*0.1)


        #ExptF=[x for x in cross_val(posx,neg,folds=folds)]
        ExptF=[x for x in sample_training(data,data_smote,label,num_train=num_train,num_test=num_test,
                                          repeats=repeats)]
        F_feature_num.append([str(n_feature)]+ExptF)
        trace_feature_num.append(n_feature)
    rdivDemo(F_feature_num)
    return {"F_feature_num": F_feature_num,"trace_feature_num": trace_feature_num}


"Train on 10,20,...,90 percent of the data, test on 10%"
@run
def train_num_change(filename='',filepath='',filetype='.txt',thres=20,repeats=30,issmote="smote",
                     neighbors=5,feature=tf,norm=l2normalize,n_range=10):
    corpus=readfile(filename=filepath+filename+filetype,thres=thres)
    #feature_num=len(vocabulary(corpus))

    feature_num=5120
    data,label=make_feature(corpus,method=feature,norm=norm,n_features=feature_num)
    if issmote=="smote":
        num=int(len(data)/len(set(label)))
        data_smote=smote(data,label,num,k=neighbors)
    else: data_smote=data
    F_train_num=[]

    num_train_init=int(len(data)*0.9)
    num_train_trace=[]
    num_test=int(len(data)*0.1)

    for i in range(1,n_range+1):
        num_train=int(i/n_range*num_train_init)
        num_train_trace.append(num_train)
        Fdistribution=[x for x in sample_training(data,data_smote,label,num_train=num_train,num_test=num_test,
                                                  repeats=repeats)]
        F_train_num.append([str(num_train)]+Fdistribution)
    rdivDemo(F_train_num)

    """
    label=[sampling.__name__+"_"+filename+".png","F1 score","Size of training set"]
    draw_curve(np.arange(0.1,1,0.1),F_train_num,label)
    """

    return {"F_train_num": F_train_num, "num_train": num_train_trace}




"Load data from file to list of lists"
def readfile(filename='',thres=20):
    corpus=[]
    targetlist=[]
    labellst=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            doc=doc.lower()
            try:
                label=doc.split(' >>> ')[1].split()[0]
                labellst.append(label)
                corpus.append([label]+doc.split(' >>> ')[0].split())
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
    F_train_num={}
    F_feature_num={}
    features=[tf,tf_idf]
    issmote=["smote","no_smote"]
    for feature in features:
        temp_train={}
        temp_feature={}
        for is_smote in issmote:


            result=train_num_change(filename=filename,filepath=filepath,filetype='.txt',
                                                                 thres=thres,repeats=30,issmote=is_smote,
                                                                 neighbors=5,feature=feature,norm=l2normalize,n_range=9)
            temp_train[is_smote]=result['F_train_num']
            trace_train_num=result['num_train']

            result=feature_num_change(filename=filename,filepath=filepath,filetype='.txt',thres=thres,
                               issmote=is_smote,neighbors=5,feature=feature,norm=l2normalize,repeats=30,n_range=9)
            temp_feature[is_smote]=result['F_feature_num']
            trace_feature_num=result['trace_feature_num']

        F_train_num[feature.__name__]=temp_train

        F_feature_num[feature.__name__]=temp_feature





    "draw"
    plt.figure(num=0,figsize=(16,12))
    plt.subplot(221)
    for feature in features:
        temp_train=F_train_num[feature.__name__]
        for is_smote in issmote:
            X=trace_train_num
            Y=temp_train[is_smote]
            Y_median=[]
            Y_iqr=[]
            for dis in Y:
                Y_median.append(np.median(dis[1:]))
                Y_iqr.append(np.percentile(dis[1:],75)-np.percentile(dis[1:],25))
            plt.plot(X,Y_median,label="median_"+feature.__name__+"_"+is_smote)
            plt.plot(X,Y_iqr,label="iqr_"+feature.__name__+"_"+is_smote)
    plt.ylabel("F score")
    plt.xlabel("Size of training set")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)




    plt.subplot(223)
    for feature in features:
        temp_feature=F_feature_num[feature.__name__]
        for is_smote in issmote:
            X=trace_feature_num
            Y=temp_feature[is_smote]
            Y_median=[]
            Y_iqr=[]
            for dis in Y:
                Y_median.append(np.median(dis[1:]))
                Y_iqr.append(np.percentile(dis[1:],75)-np.percentile(dis[1:],25))
            plt.plot(X,Y_median,label="median_"+feature.__name__+"_"+is_smote)
            plt.semilogx(X,Y_iqr,label="iqr_"+feature.__name__+"_"+is_smote)
    plt.ylabel("F score")
    plt.xlabel("Number of feature")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    plt.savefig("Zhe_"+filename+".png")

    """
    F_scott=[]
    for feature in features:
        temp_train=F_train_num[feature.__name__]
        F_scott.extend([temp_train[is_smote] for is_smote in issmote])
    rdivDemo(F_scott)
    """


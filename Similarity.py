import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import math


#MEMBUKA FILE YANG TELAH ADA
filecsv = pd.read_csv('datasetDiagnosa.csv').values.tolist()
filecsv = np.array(filecsv)
filecsv = np.ndarray.flatten(filecsv)

dataset = list(filecsv)
dataproses = list()

#PREPROCESSING
for doc in dataset:
    doc = re.sub(r'(,)|(\(([^()]+)\))', '',doc)
    dataproses.append(doc)
dataproses = np.array(dataproses).reshape(100,2)
factory = StopWordRemoverFactory()
stopwords = factory.create_stop_word_remover()
databaru = list()
for i in range(len(dataproses)):
    datapreproses = stopwords.remove(dataproses[i,0])
    datapreproses = datapreproses.lower()
    datapreproses = word_tokenize(datapreproses)
    databaru.append(datapreproses)

dataset = list()
for i in range(100):
        aray = list(np.array([databaru[i],dataproses[i,1]]))
        dataset.append(aray)

#DATA SPLIT
def Data(dataset):
    dataset = np.array(dataset)
    datatrain = dataset[:93]
    datatest = dataset[93:100]
    datatrain = np.array(datatrain)
    datatest = np.array(datatest)

    return datatrain, datatest


#TF FUNCTION
def TF(dataset):
    terms = dict()
    term = list()
    dataGejala, datatest = Data(dataset)

    test = datatest[3,0] #dapat diganti dengan angka sesuai urutan dokumen yang akan ditest
    for doc in test:
        del term[:]
        for i in range(len(dataGejala)):
            tf = dataGejala[i,0]
            itung = tf.count(doc)
            #itung = coba.count(i)
            term.append(itung)
        terms[str(doc)] = term
    return terms, dataGejala

#DF
def Df(tf_perTerm):
    df = 0
    for i in tf_perTerm:
        if i > 0:
            df += i
    return df

#IDF FUNCION
def Idf(dataset):
    idf = dict()
    tf_perTerm, datatrain = TF(dataset)

    for key,value in tf_perTerm.items():
        df = Df(tf_perTerm[key])

        idf_log = math.log((len(datatrain)/df),10) + 1
        idf[str(key)] = idf_log

    return tf_perTerm, idf

def TfIdf(dataset):
    tf_final, idf_final = Idf(dataset)
    dok_weight = list()
    key_tf = [i for i in tf_final.keys()]
    for i in range(len(tf_final[key_tf[0]])):
        weight = 0.0
        for key,value in idf_final.items():
            weight += tf_final[key][i]*value
        dok_weight.append(weight)
    print(dok_weight)
    idx_relate_dok = max(dok_weight)
    dstrain,dstest = Data(dataset)
    relate_doc = dstrain[int(idx_relate_dok),1]
    print(relate_doc)
    return relate_doc

TfIdf(dataset)

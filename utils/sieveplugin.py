# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from itertools import groupby
import pandas as pd
import torch


# +
def group_list(lst):
    return [(int(el), len(list(group))) for el, group in groupby(sorted(lst))]

def classfreqcount(det):
    clfreq = np.zeros(len(det))

    for i in range(len(det)):
        clfreq[i] = (det[i][-1])
    clfreq = clfreq.tolist()
    #print(group_list(clfreq))

    clfreq = group_list(clfreq)
    clfreq = np.array(clfreq).reshape(len(clfreq), 2)
    return clfreq

def checkiftorch(det):
    if torch.is_tensor(det):
        #print(torch.is_tensor(det))
        #print("the det matrix is a pytorch tensor and needs conversion to a numpy array")
        npdet = det.detach().to('cpu').numpy()
        return npdet
    else:
        #print(torch.is_tensor(det))
        #print("the det matrix is a numpy array")
        return det

def todf(det):
    df = pd.DataFrame(det, columns = ['x1','y1','x2','y2','x3','y3','x4','y4','conf','classidx',])
    return df


def sieve(df, nclass=4):
    max4eachclass = np.zeros((nclass,10))
    for i in range(nclass):
        temp1 = df.loc[df['classidx'] == i]
        #print(temp1)
        if not temp1.empty:
            temp2 = temp1[temp1.conf == temp1.conf.max()]
            #print(temp2)
            temp3 = temp2.to_numpy().reshape(1, 10)
            #print(temp3)
            max4eachclass[i] = temp3

    # remove rows having all zeroes
    max4eachclass = max4eachclass[~np.all(max4eachclass == 0, axis=1)]
    detsingle = torch.from_numpy(max4eachclass)
    #print(detsingle)
    return detsingle

def sievealg(det, nclass): #[i, det, ncls]
    det = checkiftorch(det)
    clfreq = classfreqcount(det)
    df = todf(det)
    maxdettorch = sieve(df, nclass)
    return maxdettorch

#---end of function definitions---

"""#--sample generation for debugging purpose--

npic = 10 # variable i in detect.py
nbox = 10 # variable det in detect.py
nclass = 4 # varaible classes in detect.py

poly = (600*np.random.rand(nbox, 8)).reshape(nbox, 8)
#print(poly.shape())
conf = np.random.rand(nbox).reshape(nbox, 1)
#print(conf.shape())
cls = np.random.randint(nclass, size=(nbox, 1)).reshape(nbox, 1)
#print(cls)

det = np.hstack((poly, conf, cls))
#conversion to torch (because according to the detect.py file, the det is a pytorch tensor)
det = torch.from_numpy(det)

#for i in range(len(det)):
#print(det[i][-1])
#print(len(det))
#print(len(det[1]))

print(det.shape)

#--end of sample generation--"""

#--function calls--

#maxdettorch = sievealg(det, nclass)

#--end of function calls--

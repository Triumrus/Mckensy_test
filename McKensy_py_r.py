
####
#№1 сортировка и вычет значения

from random import randint
 
N = 1000
a = []
for i in range(N):
    a.append(randint(1, 99))
print(a)

b=a


def quicksort(A): #Быстрая сортировка
    if A==[]:
        return A
    pivot = A.pop() #(извлечь последний или первый элемент из массива)
    lA = list(filter(lambda x: x < pivot, A)) #(создать массив с элементами меньше опорного)
    rA = list(filter(lambda x: x > pivot, A)) #(создать массив с элементами больше опорного)
    return quicksort(lA) + [pivot] + quicksort(rA) #(вернуть массив состоящий из отсортированной левой части, опорного и отсортированной правой части)

from datetime import datetime
import time
start_time = datetime.now()
z=quicksort(a)
print(datetime.now() - start_time)
#0:00:00.016000 # время исполнения быстрой сортировки

a=b


#A = [12, 5, 664, 63, 5, 73, 93, 127, 432, 64, 34,9]
def radix_sort(A):
  length = len(str(max(A)))
  rang = 10 #набор пустых значений т.е. цифры 0-9
  for i in range(length):
      B = [[] for k in range(rang)] #список длины range, состоящий из пустых списков
      for x in A:
          figure = x // 10**i % 10 # получение цифры разрядности
          B[figure].append(x)
          #print(i,x,B)
      A = []
      for k in range(rang):
          A = A + B[k]
          #print(A)
  
  return A


start_time = datetime.now()
z=radix_sort(a)
print(datetime.now() - start_time)
#0:00:00.030000 # время исполнения Поразрядная сортировка


#№2 найти в тексте все диапозонные буквы

a='babsfasffdlufwqqrkasdkflnlsdkp'

nachal=0
end=len(set(a))
qul=len(set(a))
c=0
while True:
    c+=1
    for i in sorted(set(a)):
        #print(i,i in a[nachal:end],a[nachal:end],end-nachal)
        if i not in a[nachal:end]:
            nachal+=1
            end+=1
           # print("##########")
            if end == len(a):
                nachal=0
                qul+=1
                end=qul
        elif c==len(a):
            break
    if c==len(a):
        print(end-nachal)
        break       


#№3 SQL

#№4 Мультикласс модель
# R
# library(reticulate)
# py_install("pandas")
# py_install("sklearn")
# library(reticulate)
# 
# # create a new environment 
# conda_create("r-reticulate")
# 
# # install SciPy
# conda_install("r-reticulate", "scipy")
# 
# # import SciPy (it will be automatically discovered in "r-reticulate")
# scipy <- import("scipy")
# R
import os
os.getcwd()

import pandas as pd
train=pd.read_csv("train.csv",header=None)
train.iloc[:,294:].head()
train.columns[294]
train_2 = train.iloc[:,294:]
train_2.head()
train_2.columns

train_2.groupby([train_2.columns[0],train_2.columns[1],train_2.columns[2],train_2.columns[3],train_2.columns[4],train_2.columns[5]
]).agg({
  train_2.columns[0]: 'count',train_2.columns[1]: 'count',train_2.columns[2]: 'count',train_2.columns[3]: 'count',train_2.columns[4]: 'count',train_2.columns[5]: 'count'
  })

train.shape

import random
k=[]
for i in range(train.shape[0]):
  k.append(i)

random.seed(777)
random.shuffle(k)

train=train.iloc[k,]
train.shape

train_y_label=train.iloc[:,294:]
train=train.drop([train_y_label.columns[0],train_y_label.columns[1],train_y_label.columns[2],train_y_label.columns[3],train_y_label.columns[4],train_y_label.columns[5]], axis=1)
X=train
X.head()

import numpy as np

conditions = [
    (train_y_label.iloc[:,3] == 1) & (train_y_label.iloc[:,5] == 1),
    (train_y_label.iloc[:,3] == 1) & (train_y_label.iloc[:,4] == 1),
    (train_y_label.iloc[:,2] == 1) & (train_y_label.iloc[:,4] == 1),
    (train_y_label.iloc[:,2] == 1) & (train_y_label.iloc[:,3] == 1),
    (train_y_label.iloc[:,0] == 1) & (train_y_label.iloc[:,5] == 1),
    (train_y_label.iloc[:,0] == 1) & (train_y_label.iloc[:,4] == 1),
    (train_y_label.iloc[:,5] == 1),
    (train_y_label.iloc[:,4] == 1),
    (train_y_label.iloc[:,3] == 1),
    (train_y_label.iloc[:,2] == 1),
    (train_y_label.iloc[:,1] == 1),
    (train_y_label.iloc[:,0] == 1)
    
    ]
    
choices=[1,2,3,4,5,6,7,8,9,10,11,12]
train_y_label['class'] = np.select(conditions, choices, default=0)
train_y_label.groupby('class').size()

# class
# 1       3
# 2      45
# 3       7
# 4      13
# 5      11
# 6      22
# 7     243
# 8     243
# 9     196
# 10    216
# 11    218
# 12    221
# dtype: int64


y=train_y_label['class']
y



from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from sklearn.metrics import accuracy_score

# stratificirovannay viborka
random.seed(777)
skf = StratifiedKFold(n_splits=5,shuffle=True)
l=[]
for train_index, test_index in skf.split(X, y):
    # print('train -  {}   |   test -  {}'.format(
        # np.bincount(y[train]), np.bincount(y[test])))
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y[train_index], y[test_index]
    clf=LogisticRegression(random_state=777,max_iter=5000).fit(X_train, y_train)
    l.append(accuracy_score(y_test, clf.predict(X_test)))
    print(accuracy_score(y_test, clf.predict(X_test)))

print(np.mean(l))


# it is base line but he give 0.7 ACCURACY and for this test it is enough .
# So i can stop/
# but we can continiue and up this ACCRACY

# What we can do .
# it is predict one Vs ALL
# delet fitches
# Use PCA 
accuracy_score(
y_test[0:3],
clf.predict(X_test)[0:3])








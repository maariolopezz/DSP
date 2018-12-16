
# coding: utf-8

# In[3]:


from sklearn.svm import OneClassSVM
import numpy as np 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import collections
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report, f1_score, precision_recall_fscore_support)


train = pd.read_csv('train.csv')
train = pd.concat((train, pd.read_csv('train_v2.csv')),axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('sample_submission_v2.csv')

trans = pd.read_csv('transactions.csv', usecols=['msno'])
trans = pd.concat((trans, pd.read_csv('transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
trans = pd.DataFrame(trans['msno'].value_counts().reset_index())
trans.columns = ['msno','trans_count']


train = pd.merge(train, trans, how='left', on='msno')
test = pd.merge(test, trans, how='left', on='msno')


trans = pd.read_csv('transactions_v2.csv') 
trans = trans.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
trans = trans.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, trans, how='left', on='msno')
test = pd.merge(test, trans, how='left', on='msno')
trans=[]


logs = pd.read_csv('user_logs_v2.csv', usecols=['msno'])
logs = pd.DataFrame(logs['msno'].value_counts().reset_index())
logs.columns = ['msno','logs_count']
train = pd.merge(train, logs, how='left', on='msno')
test = pd.merge(test, logs, how='left', on='msno')

logs = []; 




def get_logs(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def get_logs2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df




logs_v2 = []
logs_v2.append(get_logs(pd.read_csv('user_logs_v2.csv')))
logs_v2 = pd.concat(logs_v2, axis=0, ignore_index=True).reset_index(drop=True)
logs_v2 = get_logs2(logs_v2)
train = pd.merge(train, logs_v2, how='left', on='msno')
test = pd.merge(test, logs_v2, how='left', on='msno')
logs_v2=[]




members = pd.read_csv('members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = [];


gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

train = pd.get_dummies(train,columns=['payment_method_id'], drop_first=True)
test = pd.get_dummies(test,columns=['payment_method_id'], drop_first=True)

train = train.fillna(0)
test = test.fillna(0)




train.head()




# In[4]:


from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Lambda

from keras import regularizers
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Model, load_model



cols = [c for c in train.columns if c not in ['is_churn','msno']]

X_train = StandardScaler().fit_transform(train[cols].as_matrix())
y_train = train['is_churn'].as_matrix()
X_test = StandardScaler().fit_transform(test[cols].as_matrix())


n_hidden = 50
n_network = Sequential()

n_network.add(Dense(n_hidden, input_dim=int(X_train.shape[1]),activation='relu'))
n_network.add(BatchNormalization())
n_network.add(Dropout(rate=0.25))

n_network.add(Dense(n_hidden, activation='relu'))
n_network.add(BatchNormalization())
n_network.add(Dropout(rate=0.25))

n_network.add(Dense(n_hidden,kernel_regularizer=regularizers.l2(0.1), activation='relu'))
n_network.add(BatchNormalization())
n_network.add(Dropout(rate=0.1))

n_network.add(Dense(1, activation='sigmoid'))

n_network.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
n_network.summary()
history = n_network.fit(X_train, y_train, epochs=10, batch_size=1026,#512, 
                    validation_split=0.2, verbose=1)
results = n_network.predict(X_test)
test['is_churn'] = results.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('submission_NN.csv', index=False)





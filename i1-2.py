{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>msno</th>\n",
       "      <th>is_churn</th>\n",
       "      <th>trans_count</th>\n",
       "      <th>payment_plan_days</th>\n",
       "      <th>plan_list_price</th>\n",
       "      <th>actual_amount_paid</th>\n",
       "      <th>is_auto_renew</th>\n",
       "      <th>transaction_date</th>\n",
       "      <th>membership_expire_date</th>\n",
       "      <th>is_cancel</th>\n",
       "      <th>...</th>\n",
       "      <th>payment_method_id_32.0</th>\n",
       "      <th>payment_method_id_33.0</th>\n",
       "      <th>payment_method_id_34.0</th>\n",
       "      <th>payment_method_id_35.0</th>\n",
       "      <th>payment_method_id_36.0</th>\n",
       "      <th>payment_method_id_37.0</th>\n",
       "      <th>payment_method_id_38.0</th>\n",
       "      <th>payment_method_id_39.0</th>\n",
       "      <th>payment_method_id_40.0</th>\n",
       "      <th>payment_method_id_41.0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>waLDQMmcOu2jLDaV1ddDkgCrB/jl6sD66Xzs0Vqax1Y=</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ=</td>\n",
       "      <td>1</td>\n",
       "      <td>23</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>fGwBva6hikQmTJzrbz/2Ezjm5Cth5jZUNvXigKK2AFA=</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>mT5V8rEpa+8wuqi6x0DoVd3H5icMKkE9Prt49UlmK+4=</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>30.0</td>\n",
       "      <td>149.0</td>\n",
       "      <td>149.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>20170327.0</td>\n",
       "      <td>20170426.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>XaPhtGLk/5UvvOYHcONTwsnH97P4eGECeq+BARGItRw=</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>90.0</td>\n",
       "      <td>477.0</td>\n",
       "      <td>477.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>20170225.0</td>\n",
       "      <td>20170528.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 57 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                           msno  is_churn  trans_count  \\\n",
       "0  waLDQMmcOu2jLDaV1ddDkgCrB/jl6sD66Xzs0Vqax1Y=         1            2   \n",
       "1  QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ=         1           23   \n",
       "2  fGwBva6hikQmTJzrbz/2Ezjm5Cth5jZUNvXigKK2AFA=         1           10   \n",
       "3  mT5V8rEpa+8wuqi6x0DoVd3H5icMKkE9Prt49UlmK+4=         1            3   \n",
       "4  XaPhtGLk/5UvvOYHcONTwsnH97P4eGECeq+BARGItRw=         1            9   \n",
       "\n",
       "   payment_plan_days  plan_list_price  actual_amount_paid  is_auto_renew  \\\n",
       "0                0.0              0.0                 0.0            0.0   \n",
       "1                0.0              0.0                 0.0            0.0   \n",
       "2                0.0              0.0                 0.0            0.0   \n",
       "3               30.0            149.0               149.0            1.0   \n",
       "4               90.0            477.0               477.0            0.0   \n",
       "\n",
       "   transaction_date  membership_expire_date  is_cancel  \\\n",
       "0               0.0                     0.0        0.0   \n",
       "1               0.0                     0.0        0.0   \n",
       "2               0.0                     0.0        0.0   \n",
       "3        20170327.0              20170426.0        0.0   \n",
       "4        20170225.0              20170528.0        0.0   \n",
       "\n",
       "            ...            payment_method_id_32.0  payment_method_id_33.0  \\\n",
       "0           ...                                 0                       0   \n",
       "1           ...                                 0                       0   \n",
       "2           ...                                 0                       0   \n",
       "3           ...                                 0                       0   \n",
       "4           ...                                 0                       0   \n",
       "\n",
       "   payment_method_id_34.0  payment_method_id_35.0  payment_method_id_36.0  \\\n",
       "0                       0                       0                       0   \n",
       "1                       0                       0                       0   \n",
       "2                       0                       0                       0   \n",
       "3                       0                       0                       0   \n",
       "4                       0                       0                       0   \n",
       "\n",
       "   payment_method_id_37.0  payment_method_id_38.0  payment_method_id_39.0  \\\n",
       "0                       0                       0                       0   \n",
       "1                       0                       0                       0   \n",
       "2                       0                       0                       0   \n",
       "3                       0                       0                       0   \n",
       "4                       0                       1                       0   \n",
       "\n",
       "   payment_method_id_40.0  payment_method_id_41.0  \n",
       "0                       0                       0  \n",
       "1                       0                       0  \n",
       "2                       0                       0  \n",
       "3                       1                       0  \n",
       "4                       0                       0  \n",
       "\n",
       "[5 rows x 57 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.svm import OneClassSVM\n",
    "import numpy as np \n",
    "from sklearn.preprocessing import StandardScaler,MinMaxScaler\n",
    "import collections\n",
    "import pandas as pd\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report, f1_score, precision_recall_fscore_support)\n",
    "\n",
    "\n",
    "train = pd.read_csv('train.csv')\n",
    "train = pd.concat((train, pd.read_csv('train_v2.csv')),axis=0, ignore_index=True).reset_index(drop=True)\n",
    "test = pd.read_csv('sample_submission_v2.csv')\n",
    "\n",
    "trans = pd.read_csv('transactions.csv', usecols=['msno'])\n",
    "trans = pd.concat((trans, pd.read_csv('transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)\n",
    "trans = pd.DataFrame(trans['msno'].value_counts().reset_index())\n",
    "trans.columns = ['msno','trans_count']\n",
    "\n",
    "\n",
    "train = pd.merge(train, trans, how='left', on='msno')\n",
    "test = pd.merge(test, trans, how='left', on='msno')\n",
    "\n",
    "\n",
    "trans = pd.read_csv('transactions_v2.csv') \n",
    "trans = trans.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)\n",
    "trans = trans.drop_duplicates(subset=['msno'], keep='first')\n",
    "\n",
    "train = pd.merge(train, trans, how='left', on='msno')\n",
    "test = pd.merge(test, trans, how='left', on='msno')\n",
    "trans=[]\n",
    "\n",
    "\n",
    "logs = pd.read_csv('user_logs_v2.csv', usecols=['msno'])\n",
    "logs = pd.DataFrame(logs['msno'].value_counts().reset_index())\n",
    "logs.columns = ['msno','logs_count']\n",
    "train = pd.merge(train, logs, how='left', on='msno')\n",
    "test = pd.merge(test, logs, how='left', on='msno')\n",
    "\n",
    "logs = []; \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def get_logs(df):\n",
    "    df = pd.DataFrame(df)\n",
    "    df = df.sort_values(by=['date'], ascending=[False])\n",
    "    df = df.reset_index(drop=True)\n",
    "    df = df.drop_duplicates(subset=['msno'], keep='first')\n",
    "    return df\n",
    "\n",
    "def get_logs2(df):\n",
    "    df = df.sort_values(by=['date'], ascending=[False])\n",
    "    df = df.reset_index(drop=True)\n",
    "    df = df.drop_duplicates(subset=['msno'], keep='first')\n",
    "    return df\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "logs_v2 = []\n",
    "logs_v2.append(get_logs(pd.read_csv('user_logs_v2.csv')))\n",
    "logs_v2 = pd.concat(logs_v2, axis=0, ignore_index=True).reset_index(drop=True)\n",
    "logs_v2 = get_logs2(logs_v2)\n",
    "train = pd.merge(train, logs_v2, how='left', on='msno')\n",
    "test = pd.merge(test, logs_v2, how='left', on='msno')\n",
    "logs_v2=[]\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "members = pd.read_csv('members_v3.csv')\n",
    "train = pd.merge(train, members, how='left', on='msno')\n",
    "test = pd.merge(test, members, how='left', on='msno')\n",
    "members = [];\n",
    "\n",
    "\n",
    "gender = {'male':1, 'female':2}\n",
    "train['gender'] = train['gender'].map(gender)\n",
    "test['gender'] = test['gender'].map(gender)\n",
    "\n",
    "train = train.fillna(0)\n",
    "test = test.fillna(0)\n",
    "\n",
    "train = pd.get_dummies(train,columns=['payment_method_id'], drop_first=True)\n",
    "test = pd.get_dummies(test,columns=['payment_method_id'], drop_first=True)\n",
    "\n",
    "train = train.fillna(0)\n",
    "test = test.fillna(0)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "train.head()\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:25: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.\n",
      "/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:26: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.\n",
      "/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:27: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_1 (Dense)              (None, 50)                2800      \n",
      "_________________________________________________________________\n",
      "batch_normalization_1 (Batch (None, 50)                200       \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 50)                0         \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 50)                2550      \n",
      "_________________________________________________________________\n",
      "batch_normalization_2 (Batch (None, 50)                200       \n",
      "_________________________________________________________________\n",
      "dropout_2 (Dropout)          (None, 50)                0         \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 50)                2550      \n",
      "_________________________________________________________________\n",
      "batch_normalization_3 (Batch (None, 50)                200       \n",
      "_________________________________________________________________\n",
      "dropout_3 (Dropout)          (None, 50)                0         \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 1)                 51        \n",
      "=================================================================\n",
      "Total params: 8,551\n",
      "Trainable params: 8,251\n",
      "Non-trainable params: 300\n",
      "_________________________________________________________________\n",
      "Train on 1571112 samples, validate on 392779 samples\n",
      "Epoch 1/10\n",
      "1571112/1571112 [==============================] - 15s 10us/step - loss: 0.3440 - acc: 0.9487 - val_loss: 0.1562 - val_acc: 0.9360\n",
      "Epoch 2/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1105 - acc: 0.9585 - val_loss: 0.1497 - val_acc: 0.9349\n",
      "Epoch 3/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1090 - acc: 0.9588 - val_loss: 0.1473 - val_acc: 0.9491\n",
      "Epoch 4/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1078 - acc: 0.9590 - val_loss: 0.1535 - val_acc: 0.9378\n",
      "Epoch 5/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1071 - acc: 0.9591 - val_loss: 0.1458 - val_acc: 0.9436\n",
      "Epoch 6/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1064 - acc: 0.9592 - val_loss: 0.1510 - val_acc: 0.9383\n",
      "Epoch 7/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1059 - acc: 0.9593 - val_loss: 0.1448 - val_acc: 0.9392\n",
      "Epoch 8/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1055 - acc: 0.9594 - val_loss: 0.1550 - val_acc: 0.9247\n",
      "Epoch 9/10\n",
      "1571112/1571112 [==============================] - 14s 9us/step - loss: 0.1052 - acc: 0.9596 - val_loss: 0.1443 - val_acc: 0.9444\n",
      "Epoch 10/10\n",
      "1571112/1571112 [==============================] - 16s 10us/step - loss: 0.1050 - acc: 0.9595 - val_loss: 0.1435 - val_acc: 0.9445\n"
     ]
    }
   ],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.utils import np_utils\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.layers import Dense, Lambda\n",
    "\n",
    "from keras import regularizers\n",
    "from keras.preprocessing import sequence\n",
    "from keras.layers import Dense, Dropout, Activation\n",
    "from keras.layers import Embedding\n",
    "from keras.models import Model, load_model\n",
    "\n",
    "\n",
    "\n",
    "cols = [c for c in train.columns if c not in ['is_churn','msno']]\n",
    "\n",
    "X_train = StandardScaler().fit_transform(train[cols].as_matrix())\n",
    "y_train = train['is_churn'].as_matrix()\n",
    "X_test = StandardScaler().fit_transform(test[cols].as_matrix())\n",
    "\n",
    "\n",
    "n_hidden = 50\n",
    "n_network = Sequential()\n",
    "\n",
    "n_network.add(Dense(n_hidden, input_dim=int(X_train.shape[1]),activation='relu'))\n",
    "n_network.add(BatchNormalization())\n",
    "n_network.add(Dropout(rate=0.25))\n",
    "\n",
    "n_network.add(Dense(n_hidden, activation='relu'))\n",
    "n_network.add(BatchNormalization())\n",
    "n_network.add(Dropout(rate=0.25))\n",
    "\n",
    "n_network.add(Dense(n_hidden,kernel_regularizer=regularizers.l2(0.1), activation='relu'))\n",
    "n_network.add(BatchNormalization())\n",
    "n_network.add(Dropout(rate=0.1))\n",
    "\n",
    "n_network.add(Dense(1, activation='sigmoid'))\n",
    "\n",
    "n_network.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])\n",
    "n_network.summary()\n",
    "history = n_network.fit(X_train, y_train, epochs=10, batch_size=1026,#512, \n",
    "                    validation_split=0.2, verbose=1)\n",
    "results = n_network.predict(X_test)\n",
    "test['is_churn'] = results.clip(0.+1e-15, 1-1e-15)\n",
    "test[['msno','is_churn']].to_csv('submission_NN.csv', index=False)\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

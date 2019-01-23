from keras.layers import TimeDistributed,Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import numpy as np
import os
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score


def get_training_sequences(X_train,seq_size):

	# print(X_train[0])
	# print(X_train[1])
	till_when=len(X_train)-len(X_train)%seq_size
	take_till=X_train[:till_when]
	seqs=take_till.reshape(-1,seq_size,X_train.shape[1])

	return seqs


def normalize(X):
    return (X.astype('float32') - 120) / (X.shape[1] - 120)

	
dataset = np.load('data.npz')

X_train=dataset['X_train']
Y_train=dataset['Y_train']
X_valid=dataset['X_valid']
Y_valid=dataset['Y_valid']
X_test=dataset['X_test']
Y_test=dataset['Y_test']


X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

max_seq_size=32
num_features=115

X_train_seqs=get_training_sequences(X_train,max_seq_size)
Y_train_seqs=get_training_sequences(Y_train,max_seq_size)

X_valid_seqs=get_training_sequences(X_valid,max_seq_size)
Y_valid_seqs=get_training_sequences(Y_valid,max_seq_size)


X_test_seqs=get_training_sequences(X_valid,max_seq_size)
Y_test_seqs=get_training_sequences(Y_valid,max_seq_size)

print (X_train_seqs.shape)
print (X_valid_seqs.shape)
print(X_test_seqs.shape)

X_train_seqs = X_train_seqs.reshape(X_train_seqs.shape[0], max_seq_size, num_features, 1)
X_valid_seqs = X_valid_seqs.reshape(X_valid_seqs.shape[0], max_seq_size, num_features, 1)

model = Sequential()
model.add(TimeDistributed(Convolution1D(32, 3, activation='relu'), input_shape=(max_seq_size, num_features, 1)))
model.add(TimeDistributed(Flatten()))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True))
# model.add(LSTM(64, return_sequences=True))

model.add(TimeDistributed(Dense(12, activation='sigmoid')))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())

model.fit(X_train_seqs, Y_train_seqs,
      validation_data=(X_valid_seqs, Y_valid_seqs),
      nb_epoch=10,
      batch_size=1000,
      verbose=1
     )



Y_proba = model.predict(X_test_seqs, batch_size=batch_size, verbose=1).reshape(-1, Y_train_orig.shape[1])
Y_pred = (Y_proba >= 0.5).astype(np.int32)
Y_true=Y_test_seqs.reshape(-1, target_count)


print('accuracy:', accuracy_score(Y_true, Y_pred))
print('hamming score:', 1 - hamming_loss(Y_true, Y_pred))
print('AUC:', roc_auc_score(Y_true.flatten(), Y_proba.flatten()))

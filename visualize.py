import pickle
from extract_features import extract_features_pca,extract_features_chroma_cqt,extract_features_chroma_stft,extract_features_lda
from evaluate import evaluate_all_songs_jointly,evaluate_a_song,evaluate_classifier,evaluate_all_songs_jointly_classifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import os


def read_pkls():
	data = []
	labels = []
	files = os.listdir(os.path.join(os.getcwd(),'data'))
	files = sorted(files)
	print(files)
	for file in files:
		X = joblib.load(os.path.join(os.getcwd(),'data',file))
		data.append(X)
	files = os.listdir(os.path.join(os.getcwd(),'labels'))
	files = sorted(files)
	print(files)
	for file in files:
		Y = joblib.load(os.path.join(os.getcwd(),'labels',file))
		labels.append(Y)
	return data, labels

# def run(data, labels):
# 	evaluate_all_songs_jointly(data,labels)
# 	# evaluate_all_songs_jointly_classifier(data,labels,GaussianNB())

def visualize(X,Y):
	le=LabelEncoder()
	le.fit(Y)
	y_nums=le.transform(Y)
	# print y_nums


	X_red=extract_features_chroma_stft(X)


	tsne1=TSNE(n_components=2)


	dataConvTo2D = tsne1.fit_transform(X_red)

	x,y=dataConvTo2D.T

	plotImg=plt.figure()
	plt.scatter(x,y,c=y_nums)
	plt.show()




#Data reading after preprocessing
data, labels = read_pkls()


#Evaluation
visualize(data[0], labels[0])
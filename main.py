import pickle
from extract_features import extract_features_pca,extract_features_chroma_cqt,extract_features_chroma_stft,extract_features_lda
from evaluate import evaluate_all_songs_jointly,evaluate_a_song,evaluate_classifier,evaluate_all_songs_jointly_classifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
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

def run(data, labels):
	evaluate_all_songs_jointly(data,labels)
	# evaluate_all_songs_jointly_classifier(data,labels,GaussianNB())


#Data reading after preprocessing
data, labels = read_pkls()


#Feature extraction
data_red=[]

for i in range(len(data)):
	# song_red=extract_features_lda(data[i],labels[i])
	song_red=extract_features_chroma_cqt(data[i])

	# print song_red.shape
	# song_red=extract_features_pca(data[i])
	data_red.append(song_red)



#Evaluation
run(data_red, labels)
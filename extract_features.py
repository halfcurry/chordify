from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from librosa.feature import chroma_stft,chroma_cqt
import numpy as np

def extract_features_pca(X):
	
	pca = PCA(n_components=12)
	X_new=pca.fit_transform(X)
	return X_new

def extract_features_lda(X,Y):
	
	lda = LDA()
	X_new=lda.fit_transform(X,Y)
	return X_new


def extract_features_chroma_cqt(X,SAMPLE_RATE=22050):

	X_new=[]

	for i in range(len(X)):
		# print X[i]
		chroma_cq = chroma_cqt(y=X[i], sr=SAMPLE_RATE)
		X_new.append(chroma_cq.reshape(-1))

	return np.array(X_new)
		

def extract_features_chroma_stft(X,SAMPLE_RATE=22050):

	X_new=[]

	for i in range(len(X)):
		chroma_stft_vec = chroma_stft(y=X[i], sr=SAMPLE_RATE)
		X_new.append(chroma_stft_vec.reshape(-1))

	return np.array(X_new)




	

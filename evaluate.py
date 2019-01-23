from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def get_representative_vectors(X,Y):

	labels_to_data={}
	labels_to_Zs={}

	for i in range(len(X)):
		if Y[i] in labels_to_data:
			labels_to_data[Y[i]]=np.vstack((labels_to_data[Y[i]],X[i]))				
		else:
			labels_to_data[Y[i]]=X[i]

	for label in labels_to_data:
		# print("Shape of label vector: ", labels_to_data[label].shape)
		# labels_to_data[label].reshape(-1)
		if(labels_to_data[label].ndim > 1):
			labels_to_Zs[label]=np.mean(labels_to_data[label],axis=0)
			# print(labels_to_Zs[label].shape)
		else:
			labels_to_Zs[label]=labels_to_data[label]

	return labels_to_Zs

def get_reverse_lookup(dic):
	
	rev_lookup={}
	ctr=0
	for key in dic:
		rev_lookup[ctr]=key
		ctr+=1

	return rev_lookup


def evaluate_a_song(X,Y):

	correct=0

	X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size=0.3, random_state=42)

	# print("Size of train set: ", X_train.shape)
	# print("Size of validation set: ", X_test.shape)

	labels_to_Zs=get_representative_vectors(X_train,y_train)

	# print(labels_to_Zs)
	inds_2_labels=get_reverse_lookup(labels_to_Zs)

	preds=np.zeros(len(labels_to_Zs))
	# print()

	for j in range(len(X_test)):

		for k in range(len(preds)):

			preds[k]=np.correlate(labels_to_Zs[inds_2_labels[k]],X_test[j])

		y_h=inds_2_labels[np.argmax(preds)]		

		if y_h==y_test[j] or y_h.split(":")[0] == y_test[j].split(":")[0]:
			# print("(Predicted, label): ", y_h,y_test[j], "1")
			correct+=1
		else:
			pass
			# print("(Predicted, label): ", y_h,y_test[j], "0")

	# print (correct*100.0/len(X_test))
	return correct,len(X_test)

			
def evaluate_classifier(X,Y,clf):

			
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

	# print("Size of train set: ", X_train.shape)
	# print("Size of validation set: ", X_test.shape)
	# clf = GaussianNB()
	# clf = SVC(kernel='rbf')
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)

	correct = 0
	for j in range(len(X_test)):
		if preds[j]==y_test[j] or preds[j].split(":")[0] == y_test[j].split(":")[0]:
				# print("(Predicted, label): ", preds[j],y_test[j], "1")
				correct+=1
		else:
			pass
			# print("(Predicted, label): ", preds[j],y_test[j], "0")

	# print (correct*100.0/len(X_test))
	# print(accuracy_score(y_test,preds))

	return correct, len(X_test)

	



def evaluate_all_songs_jointly(data,labels):

	#gives RCO,ARCO,TRCO

	RCO=0
	ARCO=0
	TRCO=0	

	TRCO_num=0
	TRCO_denom=0


	for i in range(len(data)):

		correct,total=evaluate_a_song(data[i],labels[i])
		
		RCO=(correct*100.0/total)		

		ARCO+=(correct*1.0/total)

		TRCO_num+=correct
		TRCO_denom+=total

		print ("RCO for current song is :",RCO," percent.")

	TRCO=TRCO_num*1.0/TRCO_denom
	ARCO=ARCO*1.0/len(data)
	TRCO=TRCO_num*1.0/TRCO_denom


	print ("TRCO for dataset is :",TRCO)
	print ("ARCO for dataset is :",ARCO)



def evaluate_all_songs_jointly_classifier(data,labels,clf):

	#gives RCO,ARCO,TRCO

	RCO=0
	ARCO=0
	TRCO=0	

	TRCO_num=0
	TRCO_denom=0


	for i in range(len(data)):

		correct,total=evaluate_classifier(data[i],labels[i],clf)
		
		RCO=(correct*100.0/total)		

		ARCO+=(correct*1.0/total)

		TRCO_num+=correct
		TRCO_denom+=total

		print ("RCO for current song is :",RCO," percent.")

	TRCO=TRCO_num*1.0/TRCO_denom
	ARCO=ARCO*1.0/len(data)
	TRCO=TRCO_num*1.0/TRCO_denom


	print ("TRCO for dataset is :",TRCO)
	print ("ARCO for dataset is :",ARCO)



















	

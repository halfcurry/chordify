#X,Y=convert_song_to_XY(song_path,annotation_path,SAMPLE_RATE) 
#
# converts a given song and its annotation in a lab file to corresponding Xs and Ys.
# input :song_path,annotation_path,SAMPLE_RATE
#output :X= NxD extracted frames ,Y=Nx1 chord labels ('str')
#
#vrd.

import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow
import pickle
from librosa.effects import hpss


TIME_WINDOW=0.5 #seconds # 20 milliseconds
SAMPLE_RATE=22050
FRAME_WINDOW=int(TIME_WINDOW*SAMPLE_RATE)

def get_chord(cur_frame,starting_frame_to_chord):

	# print "here"
	# print cur_frame
	for pair in starting_frame_to_chord:
		if cur_frame>=pair[0] and cur_frame <=pair[1]:
			return starting_frame_to_chord[pair],pair

def get_starting_frame_to_chord(annotation_path):
	starting_frame_to_chord={}
	last_frame_annotated=0
	with open(annotation_path) as f:
		for line in f:
			line_spl=line[:-1].split(' ')
			# print line_spl
			from_sec=float(line_spl[0])
			to_sec=float(line_spl[1])
			chord_label=(line_spl[2])
			starting_frame_to_chord[(int(SAMPLE_RATE*from_sec),int(SAMPLE_RATE*to_sec))]=chord_label
			last_frame_annotated=int(SAMPLE_RATE*to_sec)

	return starting_frame_to_chord,last_frame_annotated

def get_correct_chord_label(from_frame,to_frame,starting_frame_to_chord):
	

	ch1,pair1=get_chord(from_frame,starting_frame_to_chord)		
	ch2,pair2=get_chord(to_frame,starting_frame_to_chord)

	if ch1==ch2:
		correct_chord=ch1 #(=ch2)

	elif ch1!=ch2:
		
		if (pair1[1]-from_frame) > (to_frame-pair2[0]) :
			correct_chord=ch1
		else:
			correct_chord=ch2	
	return correct_chord


			
	
def convert_song_to_XY(song_path,annotation_path,SAMPLE_RATE):
	
	# convert song to frames to feature vectors with corresponding chord labels
	
	X=[]
	Y=[]


	#open and grab info from the .mp3 and .lab files
	starting_frame_to_chord,last_frame_annotated=get_starting_frame_to_chord(annotation_path)	

	song_floats,SAMPLE_RATE=librosa.load(song_path,sr=SAMPLE_RATE)
	song_floats,_=hpss(song_floats) #Get harmonic component
	#GET Ys and Xs 


	for seek in range(0,last_frame_annotated-FRAME_WINDOW,FRAME_WINDOW):

		correct_chord=get_correct_chord_label(seek,seek+FRAME_WINDOW,starting_frame_to_chord)

		Y.append(correct_chord)
		X.append(song_floats[seek:seek+FRAME_WINDOW])	

	Y=np.array(Y)
	X=np.array(X)

	#CHECK
	# for x in xrange(len(Y)):
	# 	print X[x][0],Y[x]
	

	print(X.shape,Y.shape)


	return X,Y


def filter_main_chords(X, Y):
	X_new = []
	Y_new = []
	chords = ["C", "C:min" ,"D", "D:min", "E", "F", "F:min", "G", "G:min", "A", "A:min", "B","C#", "C#:min" ,"D", "D#:min", "F#", "F#:min", "G#", "G#:min", "A#", "A#:min"] #some standard chords
	for i,s in enumerate(X):
		if Y[i] in chords:
			X_new.append(X[i])
			Y_new.append(Y[i])

	return np.array(X_new), np.array(Y_new)

    

cur_ind=1

# convert_song_to_XY('data_curated/'+str(cur_ind)+'.mp3','data_curated/'+str(cur_ind)+'.lab',SAMPLE_RATE)

# X,Y=convert_song_to_XY('data_curated/'+str(cur_ind)+'.mp3','data_curated/'+str(cur_ind)+'.lab',SAMPLE_RATE)


# with open('X.pickle','wb') as f:
# 	pickle.dump(X,f)


# with open('Y.pickle','wb') as f:
# 	pickle.dump(Y,f)
# 	
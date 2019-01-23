import librosa, librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from preprocess_a_song import convert_song_to_XY

# TIME_WINDOW=0.5 #seconds # 500 milliseconds
SAMPLE_RATE=22050
# FRAME_WINDOW=int(TIME_WINDOW*SAMPLE_RATE)


def read_files():
	audio_files = []
	album_file_filename = {}
	song_album = {}
	file_annotation = {}

	file_to_label = {}
	i = 0
	for root, dirs, files in os.walk(os.path.join(os.getcwd(),'dataset','The Beatles')):
		for name in files:
			full_path = os.path.join(root, name)
			# print(root)
			audio_files.append(full_path)
			album = os.path.split(os.path.abspath(root))[1]
			song = os.path.split(os.path.abspath(full_path))[1]
			i+=1
			# print(i, "---", album, " ----------- ", song)
			album_num = album[:2]
			song_num = song[:2]
			# print(album_num,song_num)
			if(album_num in album_file_filename.keys()):
				album_file_filename[album_num][song_num] = full_path
			else:
				album_file_filename[album_num] = {}
				album_file_filename[album_num][song_num] = full_path

			song_album[song] = album_num


	for root, dirs, files in os.walk(os.path.join(os.getcwd(),'dataset','The Beatles Annotations')):
		for name in files:
			full_path = os.path.join(root, name)
			# print(root)
			album = os.path.split(os.path.abspath(root))[1]
			song = os.path.split(os.path.abspath(full_path))[1]
			# print(album, " ----------- ", song)
			album_num = album[:2]
			song_num = song[:2]
			file_annotation[album_file_filename[album_num][song_num]] = full_path

	return song_album, audio_files, file_annotation



# def extract_features(frame, extractor_function):
	# return extractor_function(y=frame,sr=SAMPLE_RATE)

song_album, audio_files, file_annotation = read_files()
# song = audio_files[0]
# print(song)
# print(FRAME_WINDOW)

# for song in audio_files:
# 	print(song)
# 	pass

for song in audio_files[:10]:
	name = os.path.split(os.path.abspath(song))[1]
	name = os.path.splitext(name)[0]
	if( name != '06 - Ask Me Why'):
		print(name)
		print(file_annotation[song])
		X, Y = convert_song_to_XY(song,file_annotation[song],SAMPLE_RATE)
		directory_data = os.path.join(os.getcwd(),'data')
		directory_labels = os.path.join(os.getcwd(),'labels')
		if not os.path.exists(directory_data):
			os.makedirs(directory_data)
		if not os.path.exists(directory_labels):
			os.makedirs(directory_labels)
		joblib.dump(X, os.path.join(directory_data,name+"_X.pkl"), compress=1)
		joblib.dump(Y,os.path.join(directory_labels,name+"_Y.pkl"), compress=1)

	# break

	

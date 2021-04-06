import vid_to_frame
import frame_emotion_detection
import os 
import torch
import video_bounding_box
import pandas as pd
import numpy as np
import tqdm.auto as tqdm
import zipfile

INPUT_FILE = 'Kid_mixed.mp4'
FRAME_PATH = 'KID'

CLASS_NAMES = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
#print("Unzipping models")
modellist = ['gender_detection.zip','emotion_detection_bounding_boxes.zip','emotion_detection.zip']

if os.path.exists('model'):
	os.system('rm -rf model')

for modelname in modellist:
	with zipfile.ZipFile(modelname, 'r') as zip_ref:
		zip_ref.extractall('model')
		zip_ref.close()

if __name__ == '__main__':
	feat_score_fold_0,gender_array,time_array = video_bounding_box.show_boxes(INPUT_FILE)
	#print(feat_score_fold_0)
	feat_score_fold_1 = list()
	print("Ensembling models")
	vid_to_frame.video_to_frames(video_path=INPUT_FILE, frames_dir=FRAME_PATH, overwrite=False, every=30, chunk_size=1000)
	for images in os.listdir(os.path.join(FRAME_PATH,INPUT_FILE)):
		score, codes = frame_emotion_detection.predict_emotion(os.path.join(FRAME_PATH,INPUT_FILE),images)
		#print(float(score),code)
		_,dominantcode = torch.max(codes.data, 0)
		dominantcode = dominantcode.cpu().numpy()
		#print(score[dominantcode],CLASS_NAMES[dominantcode])
		feat_score_fold_1.append(score.tolist())
	#print(feat_score_fold_1)

	feat_score_fold_1 = np.array(feat_score_fold_1)
	feat_score_fold_0 = np.array(feat_score_fold_0)
	aggregated_score = np.add(feat_score_fold_1,feat_score_fold_0)/2
	row_sum = np.sum(aggregated_score,axis=1)
	aggregated_score = (aggregated_score/row_sum[:,np.newaxis])
	print("Creating aggregated_data")
	df = pd.DataFrame(aggregated_score, columns = CLASS_NAMES)
	df['Timestamp'] = time_array
	df['Gender'] = gender_array
	df = df[df['Gender'] != 'None']
	df = df[['Timestamp','Gender','Happy','Sad','Angry','Disgusted','Fear','Surprised','Neutral']]
	print(df)
	df.to_csv('output.csv',index=False)

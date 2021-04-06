import librosa
import numpy as np 
import keras 
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(columns=['feature'])
y, sr = librosa.load('laughing.mp3')
limit = librosa.get_duration(y=y,sr=sr)
bookmark = 0

lb = LabelEncoder()
lb.fit_transform(['male_calm','female_calm','male_happy','female_happy','male_sad','female_sad','male_angry','female_angry','male_fearful','female_fearful'])
for i in np.arange(0, limit, 2.5):
	X, sample_rate = librosa.load('laughing.mp3', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=i)
	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=0)
	feature = mfccs
	#[float(i) for i in feature]
	#feature1=feature[:135]
	#print(feature)
	#print(len(feature))
	df.loc[bookmark] = [feature]
	bookmark=bookmark+1

df3 = pd.DataFrame(df['feature'].values.tolist())
df3 = df3.fillna(0)
df3 = np.expand_dims(df3, axis=2)
print(df3)

from keras.models import model_from_json
json_file = open('audio_models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("audio_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

output = loaded_model.predict(df3)
print(output)
#print(output)
print(lb.inverse_transform(output.argmax(axis=1)))

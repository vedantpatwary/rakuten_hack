from flask import Flask,flash, render_template, url_for, request
import gspread
import vid_to_frame
import frame_emotion_detection
import os 
import torch
import video_bounding_box
import pandas as pd
import numpy as np
import tqdm.auto as tqdm
import zipfile
from werkzeug.utils import secure_filename

INPUT_FILE = 'Kid_mixed.mp4'
FRAME_PATH = 'KID'

UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASS_NAMES = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
#print("Unzipping models")
modellist = ['gender_detection.zip','emotion_detection_bounding_boxes.zip','emotion_detection.zip']

if os.path.exists('model'):
    os.system('rm -rf model')

for modelname in modellist:
    with zipfile.ZipFile(modelname, 'r') as zip_ref:
        zip_ref.extractall('model')
        zip_ref.close()



@app.route("/",methods=['POST','GET'])
def index():
    if request.method == "POST":
        model_script()
        return 'Successfully pushed to server'
    else:
        return render_template("index.html")

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        # check if the post request has the file part
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            flash('File uploaded')
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            model_script(filepath)
            return 'Successfully pushed to server'
        return render_template('index.html')                       

def push_to_sheets():
    
    gc = gspread.service_account("./config/sheets_config.json")
    wks = gc.open("Emotions").sheet1
    cols = wks.col_values(1)
    rows = wks.row_values(1)
    c = len(cols)
    k = "A{}".format(c+1)
    newList = [["S4",21,"Male",0.4,0,5,0.6,0,7],["S4",21,"Male",0.4,0,5,0.6,0,7],["S4",21,"Male",0.4,0,5,0.6,0,7],["S4",21,"Male",0.4,0,5,0.6,0,7]]
    print(newList)
    wks.update(k, newList)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

def model_script(filepath):
    feat_score_fold_0,gender_array,time_array = video_bounding_box.show_boxes(filepath)
    #print(feat_score_fold_0)
    feat_score_fold_1 = list()
    print("Ensembling models")
    video_dir, video_filename = os.path.split(filepath)
    vid_to_frame.video_to_frames(video_path=filepath, frames_dir=FRAME_PATH, overwrite=True, every=30, chunk_size=1000)
    for images in os.listdir(os.path.join(FRAME_PATH,video_filename)):
        score, codes = frame_emotion_detection.predict_emotion(os.path.join(FRAME_PATH,video_filename),images)
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
    print(aggregated_score)
    df = pd.DataFrame(aggregated_score, columns = CLASS_NAMES)
    df['Timestamp'] = time_array
    df['Gender'] = gender_array
    df = df[df['Gender'] != 'None']
    df = df[['Timestamp','Gender','Happy','Sad','Angry','Disgusted','Fear','Surprised','Neutral']]
    print(df)
    df.to_csv('output.csv',index=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




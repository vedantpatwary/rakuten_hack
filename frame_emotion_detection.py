import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms
from skimage import io
from skimage.transform import resize
from models import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def predict_emotion(frame_dir,img_name):
	raw_img = io.imread(os.path.join(frame_dir,img_name))
	gray = rgb2gray(raw_img)
	gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

	img = gray[:, :, np.newaxis]

	img = np.concatenate((img, img, img), axis=2)
	img = Image.fromarray(img)
	inputs = transform_test(img)

	class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	use_cuda = torch.cuda.is_available()
	DEVICE = torch.device('cuda' if use_cuda else 'cpu')   # 'cpu' in this case

	checkpoint = torch.load(os.path.join('model', 'emotion_detection.t7'),map_location=DEVICE)

	net = VGG('VGG19')
	net.load_state_dict(checkpoint['net'])
	#net.cuda()
	net.eval()

	ncrops, c, h, w = np.shape(inputs)

	inputs = inputs.view(-1, c, h, w)
	inputs = Variable(inputs,)
	outputs = net(inputs)
	outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
	#print(outputs_avg)
	score = F.softmax(outputs_avg,dim=-1)
	_, predicted = torch.max(outputs_avg.data, 0)
	scorelist = list(score)
	return score, outputs_avg

import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import time
from matplotlib.pyplot import imread
from skimage.transform import resize

class Servo_Dataset(data.Dataset):
	def __init__(self,path,transform=None):
		self.path = path
		self.transform = transform

		#counting number o files
		self.N = len([n for n in os.listdir(osp.join(path, '.')) if
					   n.find('out') >= 0])
		print(self.N)



	def __getitem__(self, index):
		img_file0 = osp.join(self.path,'image{0:06d}_0.png'.format(index))
		img0 = self.transform(resize(imread(img_file0),(224,224)))
		return img0

	def __len__(self):
		return self.N

		


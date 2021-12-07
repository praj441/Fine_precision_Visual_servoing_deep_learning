import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import time
from matplotlib.pyplot import imread
from skimage.transform import resize

class Servo_Dataset(data.Dataset):
	def __init__(self,path,transform=None,img_size=(224,224)):
		self.path = path
		self.transform = transform
		self.size2D = img_size 

		#counting number o files
		self.N = len([n for n in os.listdir(osp.join(path, '.')) if
					   n.find('out') >= 0])
		print(self.N)

		#read poses
		self.poses = np.zeros((self.N,6))
		try:
			self.poses = np.loadtxt( osp.join(path,'all_poses.txt') , delimiter=',')
		except:
			for i in range(self.N):
			    self.poses[i] = np.loadtxt( osp.join(path,'{0:06d}.out'.format(i)) , delimiter=',')
			np.savetxt(osp.join(path,'all_poses.txt'),self.poses, delimiter=',',fmt='%1.3f')

	def __getitem__(self, index):
		img_file0 = osp.join(self.path,'image{0:06d}_0.png'.format(index))
		img_file1 = osp.join(self.path,'image{0:06d}_1.png'.format(index))
		img0 = self.transform(resize(imread(img_file0),self.size2D))
		img1 = self.transform(resize(imread(img_file1),self.size2D))
		pos = torch.from_numpy(self.poses[index]).float()
		class_label = np.array([0.0,0.0])
		if index < self.N/2 :
			if abs(pos[0:3]).mean() > 0.02 or abs(pos[3:6]).mean() > 0.009 :
				class_label[0] = 1.0
			else:
				class_label[1] = 1.0
		else:
			class_label[1] = 1.0
		class_label = torch.from_numpy(class_label).long()
		return img0,img1,pos,class_label

	def __len__(self):
		return self.N

		


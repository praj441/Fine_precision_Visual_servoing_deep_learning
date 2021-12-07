import torch 
import torch.nn as nn
from resnet1_0 import resnet18
from matplotlib.pyplot import imread
from skimage.transform import resize
import torchvision.transforms as transforms
import numpy as np
from servo_dataset_depth_only import Servo_Dataset
from torch.utils import data
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ArrayToTensor(object):
	"""Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

	def __call__(self, array):
		assert(isinstance(array, np.ndarray))
		array = np.transpose(array, (2, 0, 1))
		# handle numpy array
		tensor = torch.from_numpy(array)
		# put it from HWC to CHW format
		return tensor.float()




if __name__ == '__main__':

	if len (sys.argv) != 3 :
		print("Usage: Please provide path to the data folder and output file")
		sys.exit (1)
	train_data_path = sys.argv[1]
	mean_file_name = sys.argv[2]
	

	input_transform = transforms.Compose([
	ArrayToTensor(),
	])

	train_dataset = Servo_Dataset(train_data_path,transform=input_transform)
	train_loader = data.DataLoader(train_dataset, batch_size=1000, shuffle=True,
	num_workers=2)
	pop_mean = []
	pop_std = []
	
	for batch in train_loader:
		numpy_image = batch.numpy()
		batch_mean = np.mean(numpy_image, axis=(0,2,3))
		batch_std = np.std(numpy_image, axis=(0,2,3))
		print(batch,':',batch_mean,' ',batch_std)
		pop_mean.append(batch_mean)
		pop_std.append(batch_std)
		
	pop_mean = np.array(pop_mean).mean(axis=0)
	pop_std = np.array(pop_std).mean(axis=0)
	mean_file_content = np.zeros((6,))
	mean_file_content[0:3] = pop_mean
	mean_file_content[3:6] = pop_std
	np.savetxt(mean_file_name,mean_file_content, delimiter=',',fmt='%1.3f')
	print('mean:',pop_mean)
	print('std:',pop_std)
		
	
			
	

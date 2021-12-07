import torch 
import torch.nn as nn
from resnet1_0 import resnet18
from matplotlib.pyplot import imread
from skimage.transform import resize
import torchvision.transforms as transforms
import numpy as np
from servo_dataset_implicit import Servo_Dataset
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

class rank_pool(nn.Module):
	def __init__(self,in_planes, out_planes, size):
		super(rank_pool, self).__init__()
		self.exp = 4 # Expension Ratio (E)
		self.p1 = nn.Conv2d(in_planes, out_planes*self.exp, 3, 2, 1, groups=in_planes, bias=False) # DepthWise Convolution using 3x3 kernel
		self.bn1 = nn.BatchNorm2d(out_planes*self.exp)

		self.p2 = nn.Conv2d(out_planes*self.exp, out_planes, kernel_size=1, stride=1, padding=0, bias=False) # PointWise Convolution using 1x1 kernel
		self.bn2 = nn.BatchNorm2d(out_planes)

		self.p3 = nn.Conv2d(out_planes, out_planes, size, 1, 0, groups=out_planes, bias=False) # Ranking using DepthWise Convolution (WxH kernel) 
		self.bn3 = nn.BatchNorm2d(out_planes) 
		self.sig = nn.Sigmoid()
	
	def forward(self, x):
		out = F.relu(self.bn1(self.p1(x)))
		out = self.bn2(self.p2(out))

		y = self.sig(self.bn3(self.p3(out)))
		out = out*y.view(x.size(0),-1,1,1) 
		out = F.relu(out)       
		return out
 
#********************* NETWORK DESIGN ****************************************

class VSNet(nn.Module):
	

	def __init__(self,batchNorm=True):
		super(VSNet,self).__init__()
		self.batchNorm = batchNorm
		self.sx = nn.Parameter(torch.tensor(0.0))
		self.sq = nn.Parameter(torch.tensor(-3.0))
				
		self.base_cnn = resnet18(pretrained=True)  # Only till top 5 CNN layers are used
		self.conv_block = nn.Sequential(
			nn.Conv2d(128,256, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			rank_pool(256,256,28),
			nn.Conv2d(256,512, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			#nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			rank_pool(512,512,14),
			nn.Conv2d(512,1024, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(1024),
			nn.ReLU(inplace=True),
			#nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			rank_pool(1024,1024,7),
			nn.AdaptiveAvgPool2d((1, 1))
			)

		self.fc_feature = nn.Linear(1024,4096)
		self.fc = nn.Linear(4096,6)
		self.fc_cls1 = nn.Linear(1024,256)
		self.fc_cls2 = nn.Linear(256,2)

	def forward(self, x1,x2):
		v1 = self.base_cnn(x1)
		v2 = self.base_cnn(x2)
		v = torch.cat((v1,v2),dim=1)
		v = self.conv_block(v)
		v = v.view(v.size(0), -1)
		rg = self.fc_feature(v)
		out_rg = self.fc(rg)
		#Auxiliery classiifcation head
		cl = self.fc_cls1(v)
		cl = self.fc_cls2(cl)
		out_cl = F.softmax(cl,dim=1)
		return out_rg,out_cl

	# function for auto balance the losses
	def learned_weighting_loss(self, loss_pos, loss_rot):
		'''The weighted loss function that learns variables sx and sy to balance the positional loss and the rotational loss'''
		return (-1 * self.sx).exp() * loss_pos + self.sx + (-1 * self.sq).exp() * loss_rot + self.sq

def cls_accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = torch.argmax(output, dim=1) 
    truth = torch.argmax(target,dim=1) 
    acc = pred.eq(truth).sum().item() / truth.numel()
    return acc

#********************* MAIN FUNCTION FOR TRAINING *****************************

if __name__ == '__main__':

	if len (sys.argv) != 4 :
		print("Usage: Please provide a h_list file hyperparameters values ,mean file for mean values and a version name")
		sys.exit (1)
	hp_file = sys.argv[1]
	mean_file = sys.argv[2]
	version = sys.argv[3]
	#loading hyperparameters from files 
	try:
		with open(hp_file, "r") as fp:   # Unpickling
			hp = eval(fp.readline())
			print('hyperparameters loaded')

	except:
		hp = [50, 0.00001, 5e-04, '../data/scale_match_RO_lsd/train', '../data/scale_match_RO_lsd/val', [0],70, 0,3,6,0.2,False,'trained_nets/model_weights.pt']
		fp.write(str(hp))
		print('hyperparameters dumped')

	batch_size = hp[0]
	lr = hp[1]
	weight_decay = hp[2]
	train_data_path = hp[3]
	val_data_path = hp[4]
	cuda_devices = hp[5]
	epochs = hp[6]
	n1 = hp[7]
	n2 = hp[8]
	n3 = hp[9]
	B = hp[10]
	load_pretrained_vs = hp[11]
	pretrained_file = hp[12]

	try:
		mean_file_content = np.loadtxt( mean_file , delimiter=',')
		mean = mean_file_content[0:3]
		std = mean_file_content[3:6]
		print('mean and std loaded from file ',mean_file)
		print('mean:',mean)
		print('std:',std)

	except:
		mean = [0.44443277, 0.44384047, 0.44420946]
		std = [0.2450024 , 0.2455248 , 0.24519353]
	

	input_transform = transforms.Compose([
	ArrayToTensor(),
	transforms.Normalize(mean=mean, std=std)
	])

	train_dataset = Servo_Dataset(train_data_path,transform=input_transform)
	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
	num_workers=2)

	val_dataset = Servo_Dataset(val_data_path,transform=input_transform)
	val_total = val_dataset.__len__()
	val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
	num_workers=2)

	model = VSNet().to(device)
	
	if load_pretrained_vs:
		model.load_state_dict(torch.load(pretrained_file))
		print("loaded pretrained VSNet:",pretrained_file)
	model = torch.nn.DataParallel(model, device_ids=cuda_devices)
	
	loss1 = torch.nn.CrossEntropyLoss()
	loss2 = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)

	for name,param in model.named_parameters():
		print(name,param.requires_grad)

	#epochs = 80
	min_loss = 10
	train_loss_arr = np.zeros((epochs,1))
	val_loss_arr = np.zeros((epochs,1))
	offset = torch.Tensor([2]).float().to(device)
	for i in range(0,epochs):
		if epochs==30:
			for param_group in optimizer.param_groups:
				param_group['lr'] = 0.00001

		start = time.time()

#****************************************calculating loss over train subset****************************
		total_loss_train = 0
		total_loss_train1 = 0
		total_loss_train2 = 0
		total_ac = 0
		batch_count = 0
		label = None
		out = None
		for batch in train_loader:
			model.eval()
			batch_count += 1
			if batch_count > 20:
				break
			#extracting input and label
			img0 = batch[0].to(device)
			img1 = batch[1].to(device)
			label = batch[2].to(device)
			label_c = batch[3].to(device)
			#forward pass
			out_r,out_c = model(img0,img1)
			#Classiifcation_loss
			total_ac += cls_accuracy(out_c,label_c)
			loss_b1 = loss2(out_r[:,n1:n2],label[:,n1:n2])
			loss_b2 = loss2(out_r[:,n2:n3],label[:,n2:n3])
			total_loss_train1 += loss_b1.item()
			total_loss_train2 += loss_b2.item()
		train_ac = total_ac / batch_count
		avg_loss_train1 = total_loss_train1 / batch_count
		avg_loss_train2 = total_loss_train2 / batch_count
		avg_loss_train = (avg_loss_train1 + B * avg_loss_train2 ) / (1+B)
		if(avg_loss_train < min_loss):
			min_loss = avg_loss_train
		train_loss_arr[i] = avg_loss_train
#****************************************************************************************************

#****************************************calculating loss over val subset****************************

		total_ac = 0
		total_loss_val = 0
		total_loss_val1 = 0
		total_loss_val2 = 0
		batch_count = 0
		for batch in val_loader:
			model.eval()
			batch_count += 1
			if batch_count > 10:
				break
			#extracting input and label
			img0 = batch[0].to(device)
			img1 = batch[1].to(device)
			label = batch[2].to(device)
			label_c = batch[3].to(device)
			#forward pass
			out_r,out_c = model(img0,img1)
			#Classiifcation_loss
			total_ac += cls_accuracy(out_c,label_c)
			loss_b1 = loss2(out_r[:,n1:n2],label[:,n1:n2])
			loss_b2 = loss2(out_r[:,n2:n3],label[:,n2:n3])
			total_loss_val1 += loss_b1.item()
			total_loss_val2 += loss_b2.item()

		val_ac = total_ac / (batch_count)
		avg_loss_val1 = total_loss_val1 / batch_count
		avg_loss_val2 = total_loss_val2 / batch_count
		avg_loss_val = (avg_loss_val1 + B * avg_loss_val2) / (1+B)
		val_loss_arr[i] = val_ac

		print('epoch:',i,'val_loss(Tr,Rot,Overall):',avg_loss_val1,avg_loss_val2,avg_loss_val,'train_loss(Tr,Rot,Overall):',avg_loss_train1,avg_loss_train2,avg_loss_train)
		print('Classifcation_accuracy--','train:',train_ac,'validation:',val_ac)
		torch.save(model.module.state_dict(),'trained_nets/{0}_{1}.pt'.format(version,i))
		# ************ Printing Few GroundTruth labels and model prediction for visualization************
		print('label',label[0:5,:])
		print('out',out_r[0:5,:])
#****************************************************************************************************


#****************************************Training one epoch*******************************************

		batch_count = 0
		avg_loss_cl = 0
		total_loss_cl = 0
		for batch in train_loader:
			model.train()
			optimizer.zero_grad()
			batch_count += 1
			#print('train_batch:',batch_count)
			if batch_count == 2000:
				break;
			if batch_count%20==0:
				print("training-","epoch:",i,"batch:",batch_count)
			#extracting input and label
			img0 = batch[0].to(device)
			img1 = batch[1].to(device)
			label = batch[2].to(device)
			label_c = batch[3].to(device)
			#forward pass
			out_r,out_c = model(img0,img1)
			#loss
			loss_cl = loss1(out_c,torch.max(label_c, 1)[1])
			loss_b = loss2(out_r[:,n1:n2],label[:,n1:n2]) + B * loss2(out_r[:,n2:n3],label[:,n2:n3])  

			#backward pass
			loss = model.module.learned_weighting_loss(loss_cl,loss_b)
			loss.backward()
			optimizer.step()

			total_loss_cl += loss_cl
		avg_loss_cl = total_loss_cl/batch_count
#****************************************************************************************************
			
		np.savetxt('train_loss_{0}.txt'.format(version),train_loss_arr, delimiter=',',fmt='%.6f')
		np.savetxt('val_loss_{0}.txt'.format(version),val_loss_arr, delimiter=',',fmt='%.6f')
		print('Time for the epoch:',(time.time()-start))
	print('min_loss=',min_loss)
	plt.plot(train_loss_arr,label='train')
	plt.plot(val_loss_arr,label='val')
	plt.legend()
	plt.savefig('plot_{0}.png'.format(version))
	

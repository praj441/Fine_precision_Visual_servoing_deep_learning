import torch 
import torch.nn as nn
from resnet1_0 import resnet18
from matplotlib.pyplot import imread
from skimage.transform import resize
import torchvision.transforms as transforms
import numpy as np

from servo_dataset_cls import Servo_Dataset_Cls
from servo_dataset_cls1 import Servo_Dataset_Cls1
from servo_dataset_cls2 import Servo_Dataset_Cls2
from servo_dataset import Servo_Dataset
from servo_dataset_1st_half import Servo_Dataset1
from servo_dataset_2nd_half import Servo_Dataset2
from torch.utils import data
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from copy import deepcopy
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
		self.s1 = nn.Parameter(torch.tensor(-6.0))
		self.s2 = nn.Parameter(torch.tensor(-12.0))
		self.s3 = nn.Parameter(torch.tensor(0.0))

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

		self.fc_feature_lsd = nn.Linear(1024,4096)
		self.fc_lsd = nn.Linear(4096,6)
		self.fc_feature_ssd = nn.Linear(1024,4096)
		self.fc_ssd = nn.Linear(4096,6)
		self.fc_cls = nn.Linear(1024,2)

	def forward(self, x1,x2):
		v1 = self.base_cnn(x1)
		v2 = self.base_cnn(x2)
		v = torch.cat((v1,v2),dim=1)
		v = self.conv_block(v)
		v = v.view(v.size(0), -1)
		rgl = self.fc_feature_lsd(v)
		out_rgl = self.fc_lsd(rgl)
		rgs = self.fc_feature_ssd(v)
		out_rgs = self.fc_ssd(rgs)
		cl = self.fc_cls(v)
		out_cl = F.softmax(cl,dim=1)#softmax for auxiliery classiifcation head
		return out_rgl,out_rgs,out_cl

	def learned_weighting_loss(self, loss1, loss2,loss3):
		'''The weighted loss function that learns variables sx and sy to balance the positional loss and the rotational loss'''
		return (-1 * self.s1).exp() * loss1 + self.s1 + (-1 * self.s2).exp() * loss2 + self.s2 + (-1 * self.s3).exp() * loss3 + self.s3

	def load_my_state_dict(self, state_dict):
		own_state = self.state_dict()
		for name, param in state_dict.items():
			if name not in own_state:
				 continue
			own_state[name].copy_(param.data)

def cls_accuracy(output, target):
    pred = torch.argmax(output, dim=1) 
    truth = torch.argmax(target,dim=1) 
    acc = pred.eq(truth).sum().item() / truth.numel()
    return acc

#********************* MAIN FUNCTION FOR TRAINING *****************************

if __name__ == '__main__':

	if len (sys.argv) != 4 :
		print("Usage: Please provide a h_list file hyperparameters values, mean file for mean values and a version name")
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

	train_dataset1 = Servo_Dataset_Cls1('../data/combine/train',transform=input_transform)
	train_dataset2 = Servo_Dataset1('../data/scale_match_RO_lsd/train',transform=input_transform)
	train_dataset3 = Servo_Dataset1('../data/scale_match_RO_ssd/train',transform=input_transform)
	train_loader1 = data.DataLoader(train_dataset1, batch_size=batch_size, shuffle=True,
	num_workers=2)
	train_loader2 = data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=True,
	num_workers=2)
	train_loader3 = data.DataLoader(train_dataset3, batch_size=batch_size, shuffle=True,
	num_workers=2)

	meta_dataset1 = Servo_Dataset_Cls2('../data/combine/train',transform=input_transform)
	meta_dataset2 = Servo_Dataset2('../data/scale_match_RO_lsd/train',transform=input_transform)
	meta_dataset3 = Servo_Dataset2('../data/scale_match_RO_ssd/train',transform=input_transform)
	meta_loader1 = data.DataLoader(meta_dataset1, batch_size=batch_size, shuffle=True,
	num_workers=2)
	meta_loader2 = data.DataLoader(meta_dataset2, batch_size=batch_size, shuffle=True,
	num_workers=2)
	meta_loader3 = data.DataLoader(meta_dataset3, batch_size=batch_size, shuffle=True,
	num_workers=2)

	val_dataset1 = Servo_Dataset_Cls('../data/combine/val',transform=input_transform)
	val_dataset2 = Servo_Dataset('../data/scale_match_RO_lsd/val',transform=input_transform)
	val_dataset3 = Servo_Dataset('../data/scale_match_RO_ssd/val',transform=input_transform)
	val_loader1 = data.DataLoader(val_dataset1, batch_size=batch_size, shuffle=True,
	num_workers=2)
	val_loader2 = data.DataLoader(val_dataset2, batch_size=batch_size, shuffle=True,
	num_workers=2)
	val_loader3 = data.DataLoader(val_dataset3, batch_size=batch_size, shuffle=True,
	num_workers=2)

	model = VSNet().to(device)
	
	if load_pretrained_vs:
		model.load_my_state_dict(torch.load(pretrained_file))
		print("loaded pretrained VSNet:",pretrained_file)
	model = torch.nn.DataParallel(model, device_ids=cuda_devices)
	
	loss1 = torch.nn.CrossEntropyLoss()
	loss2 = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)
	optimizer_meta = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=weight_decay)

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
		total_loss_train3 = 0
		total_loss_train31 = 0
		total_loss_train32 = 0
		total_loss_train2 = 0
		total_loss_train21 = 0
		total_loss_train22 = 0
		total_ac = 0
		batch_count = 0
		label = None
		out = None
		iter2 = iter(train_loader2)
		iter3 = iter(train_loader3)
		for batch1 in train_loader1:
			model.eval()
			batch_count += 1
			batch2 = next(iter2)
			batch3 = next(iter3)
			if batch_count > 20:
				break
			#extracting input and label
			img10 = batch1[0].to(device)
			img11 = batch1[1].to(device)
			label1 = batch1[2].to(device)
			img20 = batch2[0].to(device)
			img21 = batch2[1].to(device)
			label2 = batch2[2].to(device)
			img30 = batch3[0].to(device)
			img31 = batch3[1].to(device)
			label3 = batch3[2].to(device)
			#forward pass
			_,_,out_c = model(img10,img11)
			out_rl,_,_ = model(img20,img21)
			_,out_rs,_ = model(img30,img31)
			#Classiifcation_loss
			total_ac += cls_accuracy(out_c,label1)
			loss_b21 = loss2(out_rl[:,n1:n2],label2[:,n1:n2])
			loss_b22 = loss2(out_rl[:,n2:n3],label2[:,n2:n3])
			total_loss_train21 += loss_b21.item()
			total_loss_train22 += loss_b22.item()
			loss_b31 = loss2(out_rs[:,n1:n2],label3[:,n1:n2])
			loss_b32 = loss2(out_rs[:,n2:n3],label3[:,n2:n3])
			total_loss_train31 += loss_b31.item()
			total_loss_train32 += loss_b32.item()
		train_ac = total_ac / batch_count
		avg_loss_train21 = total_loss_train21 / batch_count
		avg_loss_train22 = total_loss_train22 / batch_count
		avg_loss_train2 = (avg_loss_train21 + B * avg_loss_train22 ) / (1+B)
		train_loss_arr[i] = avg_loss_train2

		avg_loss_train31 = total_loss_train31 / batch_count
		avg_loss_train32 = total_loss_train32 / batch_count
		avg_loss_train3 = (avg_loss_train31 + B * avg_loss_train32 ) / (1+B)
#****************************************************************************************************

#****************************************calculating loss over val subset****************************

		total_loss_val3 = 0
		total_loss_val31 = 0
		total_loss_val32 = 0
		total_loss_val2 = 0
		total_loss_val21 = 0
		total_loss_val22 = 0
		total_ac = 0
		batch_count = 0
		label = None
		out = None
		iter2 = iter(val_loader2)
		iter3 = iter(val_loader3)
		model.eval()
		for batch1 in val_loader1:
			batch_count += 1
			batch2 = next(iter2)
			batch3 = next(iter3)
			if batch_count > 18:
				break
			#extracting input and label
			img10 = batch1[0].to(device)
			img11 = batch1[1].to(device)
			label1 = batch1[2].to(device)
			img20 = batch2[0].to(device)
			img21 = batch2[1].to(device)
			label2 = batch2[2].to(device)
			img30 = batch3[0].to(device)
			img31 = batch3[1].to(device)
			label3 = batch3[2].to(device)
			#forward pass
			_,_,out_c = model(img10,img11)
			out_rl,_,_ = model(img20,img21)
			_,out_rs,_ = model(img30,img31)
			#Classiifcation_loss
			total_ac += cls_accuracy(out_c,label1) #+ B * loss(out[:,n2:n3],label[:,n2:n3])    #old loss function
			loss_b21 = loss2(out_rl[:,n1:n2],label2[:,n1:n2])
			loss_b22 = loss2(out_rl[:,n2:n3],label2[:,n2:n3])
			total_loss_val21 += loss_b21.item()
			total_loss_val22 += loss_b22.item()

			loss_b31 = loss2(out_rs[:,n1:n2],label3[:,n1:n2])
			loss_b32 = loss2(out_rs[:,n2:n3],label3[:,n2:n3])
			total_loss_val31 += loss_b31.item()
			total_loss_val32 += loss_b32.item()
		val_ac = total_ac / batch_count
		avg_loss_val21 = total_loss_val21 / batch_count
		avg_loss_val22 = total_loss_val22 / batch_count
		avg_loss_val2 = (avg_loss_val21 + B * avg_loss_val22 ) / (1+B)
		val_loss_arr[i] = val_ac

		avg_loss_val31 = total_loss_val31 / batch_count
		avg_loss_val32 = total_loss_val32 / batch_count
		avg_loss_val3 = (avg_loss_val31 + B * avg_loss_val32 ) / (1+B)

		print('epoch:',i)
		print('lsd:','val_loss:',avg_loss_val21,'+B*',avg_loss_val22,'=',avg_loss_val2,'train_loss:',avg_loss_train21,'+B*',avg_loss_train22,'=',avg_loss_train2,'time:',(time.time()-start))
		print('ssd:','val_loss:',avg_loss_val31,'+B*',avg_loss_val32,'=',avg_loss_val3,'train_loss:',avg_loss_train31,'+B*',avg_loss_train32,'=',avg_loss_train3,'time:',(time.time()-start))

		print('Classifcation_accuracy--','train:',train_ac,'validation:',val_ac)
		print('balancing_weights:',(-1*model.module.s1).exp().item(),(-1*model.module.s2).exp().item(),(-1*model.module.s3).exp().item())
		torch.save(model.module.state_dict(),'trained_nets/mytraining_train_overfit{0}_{1}_{2}.pt'.format(version,i,val_ac))

		print('printing some samples for large scale data(LSD)')
		print('label',label2[0:5,n1:n3])
		print('out',out_rl[0:5,n1:n3])

		print('printing some samples for short scale data(LSD)')
		print('label',label3[0:5,n1:n3])
		print('out',out_rs[0:5,n1:n3])
#****************************************************************************************************

	
#****************************************Training one epoch*******************************************
		torch.autograd.set_detect_anomaly(True)
		batch_count = 0
		avg_loss_cl = 0
		total_loss_cl = 0
		iter2 = iter(train_loader2)
		iter3 = iter(train_loader3)
		meta_iter1 = iter(meta_loader1)
		meta_iter2 = iter(meta_loader2)
		meta_iter3 = iter(meta_loader3)
		model1 = VSNet().to(device)
		model1 = torch.nn.DataParallel(model1, device_ids=cuda_devices)
		model2 = VSNet().to(device)
		model2 = torch.nn.DataParallel(model2, device_ids=cuda_devices)
		model3 = VSNet().to(device)
		model3 = torch.nn.DataParallel(model3, device_ids=cuda_devices)
		model_temp = VSNet().to(device)
		model_temp = torch.nn.DataParallel(model_temp, device_ids=cuda_devices)
		optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=lr,weight_decay=weight_decay) # get new optimiser
		optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr,weight_decay=weight_decay) # get new optimiser
		optimizer3 = torch.optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=lr,weight_decay=weight_decay) # get new optimiser
		
		model.train()
		model1.train()
		model2.train()
		model3.train()
		for batch1 in train_loader1:
			batch_count += 1
			batch2 = next(iter2)
			batch3 = next(iter3)
			#extracting input and label
			img10 = batch1[0].to(device)
			img11 = batch1[1].to(device)
			label1 = batch1[2].to(device)
			img20 = batch2[0].to(device)
			img21 = batch2[1].to(device)
			label2 = batch2[2].to(device)
			img30 = batch3[0].to(device)
			img31 = batch3[1].to(device)
			label3 = batch3[2].to(device)

			model1.load_state_dict(model.state_dict())
			_,_,out_c = model1(img10,img11)
			optimizer1.load_state_dict(optimizer.state_dict())
			optimizer1.zero_grad()
			loss_cl = loss1(out_c,torch.max(label1, 1)[1])
			loss_cl.backward()
			optimizer1.step()

			model2.load_state_dict(model.state_dict())
			out_rl,_,_ = model2(img20,img21)
			optimizer2.load_state_dict(optimizer.state_dict())
			optimizer2.zero_grad()
			loss_b2 = loss2(out_rl[:,n1:n2],label2[:,n1:n2]) + B * loss2(out_rl[:,n2:n3],label2[:,n2:n3])  
			loss_b2.backward()
			optimizer2.step()

			model3.load_state_dict(model.state_dict())
			_,out_rs,_ = model3(img30,img31)
			optimizer3.load_state_dict(optimizer.state_dict())
			optimizer3.zero_grad()
			loss_b3 = loss2(out_rs[:,n1:n2],label3[:,n1:n2]) + B * loss2(out_rs[:,n2:n3],label3[:,n2:n3])  
			loss_b3.backward()
			optimizer3.step()
			############################## meta-update ########################
			batch1 = next(meta_iter1)
			batch2 = next(meta_iter2)
			batch3 = next(meta_iter3)
			img10 = batch1[0].to(device)
			img11 = batch1[1].to(device)
			label1 = batch1[2].to(device)
			img20 = batch2[0].to(device)
			img21 = batch2[1].to(device)
			label2 = batch2[2].to(device)
			img30 = batch3[0].to(device)
			img31 = batch3[1].to(device)
			label3 = batch3[2].to(device)

			model_temp.load_state_dict(model.state_dict())

			model.load_state_dict(model1.state_dict())
			_,_,out_c = model(img10,img11)
			loss_cl = loss1(out_c,torch.max(label1, 1)[1])

			model.load_state_dict(model2.state_dict())
			out_rl,_,_ = model(img20,img21)
			loss_b2 = loss2(out_rl[:,n1:n2],label2[:,n1:n2]) + B * loss2(out_rl[:,n2:n3],label2[:,n2:n3])  

			model.load_state_dict(model3.state_dict())
			_,out_rs,_ = model(img30,img31)
			loss_b3 = loss2(out_rs[:,n1:n2],label3[:,n1:n2]) + B * loss2(out_rs[:,n2:n3],label3[:,n2:n3])  
			
			model.load_state_dict(model_temp.state_dict())
			meta_loss = model.module.learned_weighting_loss(loss_b2,loss_b3,loss_cl)
			optimizer.zero_grad()
			meta_loss.backward()
			optimizer.step()

			total_loss_cl += loss_cl

			if batch_count%20==0:
				print("training-","epoch:",i,"batch:",batch_count,"meta_loss:",meta_loss.item())
			if batch_count == 1000:
				break;
#****************************************************************************************************
			
		avg_loss_cl = total_loss_cl/batch_count
		np.savetxt('plots/train_loss_{0}.txt'.format(version),train_loss_arr, delimiter=',',fmt='%.6f')
		np.savetxt('plots/val_loss_{0}.txt'.format(version),val_loss_arr, delimiter=',',fmt='%.6f')
		print('cls_train_loss=',avg_loss_cl)
	plt.plot(train_loss_arr,label='train')
	plt.plot(val_loss_arr,label='val')
	plt.legend()
	plt.savefig('plots/plot_{0}.png'.format(version))
	

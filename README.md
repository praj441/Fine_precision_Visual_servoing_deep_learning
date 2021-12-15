# Fine_precision_Visual_servoing_deep_learning
Implementation of the paper "Learning to Switch CNNs with Model Agnostic Meta Learning for Fine Precision Visual Servoing." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020."

**Abtract:**
Convolutional Neural Networks (CNNs) have been successfully applied for relative camera pose estimation from labeled image-pair data, without requiring any handengineered features, camera intrinsic parameters or depth information. The trained CNN can be utilized for performing pose based visual servo control (PBVS). One of the ways to improve the quality of visual servo output is to improve the accuracy of the CNN for estimating the relative pose estimation. With a given state-of-the-art CNN for relative pose regression, how can we achieve an improved performance for visual servo control? In this paper, we explore switching of CNNs to improve the precision of visual servo control. The idea of switching a CNN is due to the fact that the dataset for training a relative camera pose regressor for visual servo control must contain variations in relative pose ranging from a very small scale to eventually a larger scale. We found that, training two different instances of the CNN, one for large-scale-displacements (LSD) and another for small-scale-displacements (SSD) and switching them during the visual servo execution yields better results than training a single CNN with the combined LSD+SSD data. However, it causes extra storage overhead and switching decision is taken by a manually set threshold which may not be optimal for all the scenes. To eliminate these drawbacks, we propose an efficient switching strategy based on model agnostic meta learning (MAML) algorithm. In this, a single model is trained to learn parameters which are simultaneously good for multiple tasks, namely a binary classification for switching decision, a 6DOF pose regression for LSD data and also a 6DOF pose regression for SSD data. The proposed approach performs far better than the naive approach, while storage and run-time overheads are almost negligible.

**For any query please contact** premr441@gmail.com  or raise an issue here.

**Validation data** to verify the different parts of the code - [Download](https://drive.google.com/file/d/1YPyqyM98L4PS2BcXUs3rVLRNFtU10aye/view?usp=sharing) Size:178.5 MB. It contains two different folder named lsd(for large scale displacements) and ssd(short scale displacements). Each folder contain 1000 data samples. 

**Training data** - [Download](https://www.dropbox.com/s/tkqfty4yx6lf3jn/train_data_vs1.zip?dl=0) size:35.5 GB. One single zip file. This file also contain the validation data.

**Trained weights** - [Download here](https://drive.google.com/file/d/1MNo3KuChqkdwvjiRsE4JiVhTaKVA07BZ/view?usp=sharing) Size:198.3 MB. Currently uploaded trained weights for 4 variants which are as follow:-

lsd.pt - model trained with lsd data (100K data samples)

ssd.pt - model trained with ssd data (100K data samples)

combine.pt - model trained with lsd+ssd data (200K data samples)

combine_switch - model trained with lsd+ssd data using implicit switch technique (200K data samples)

*Other weights can be made available on request basis. 

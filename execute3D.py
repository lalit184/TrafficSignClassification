from __future__ import print_function
from argparse import ArgumentParser
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from parameters import params

from keras.preprocessing import image
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import pylab as plt

import numpy as np
import keras
import h5py 
import matplotlib.pyplot as plt
import os
import csv
import scipy.misc
from model import *
import tensorflow as tf

#ros imports 
import cv2
from sensor_msgs.msg import Image
# from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge,CvBridgeError
import rospy

bridge = CvBridge()

class Resnet(params):
	def __init__(self,condition):
		super(Resnet, self).__init__()
		if condition == "training":
			self.model = Classifier(self.input_tensor,self.n_classes)
			
			for layer in self.model.layers:
		   		layer.trainable = True
		else:
			self.model = load_model('./check_point/weights-04-0.05.h5')
			self.model._make_predict_function()
			
		
		
	def	train(self):
		"""
		This function as the name suggests is for training the neural net
		there is an augmentation that the user can use if they wish to
		after training is finished as per the traininng finishing criterion iff satisfied by 
		early stopper the functions plots the loss curve and val loss curve
		"""
		self.X_train, self.Y_train, self.X_val, self.Y_val= self.images_and_labels("training")
		
		if self.data_augmentation:
			print('Using real-time data augmentation.')
			
			self.datagen.fit(self.X_train)
			print("augmented_data generated. now fitting the model")
			
			self.run_model = self.model.fit_generator(self.datagen.flow(self.X_train, self.Y_train, batch_size=self.batch_size),steps_per_epoch=self.X_train.shape[0] // self.batch_size,validation_data=(self.X_val, self.Y_val),epochs=self.nb_epoch, verbose=1, max_q_size=100,callbacks=[self.save_after_epoch,self.csv])
		else:
			print('Not using data augmentation.')
			self.run_model = self.model.fit(self.X_train, self.Y_train,batch_size = self.batch_size,nb_epoch = self.nb_epoch,validation_data=(self.X_val, self.Y_val),shuffle=True,callbacks=[self.save_after_epoch,self.csv,self.early_stopper],verbose = 1)
		
		self.plot_all()
		self.model.save(self.save_dir)

	def get_statistics(self,results,GT):

		"""
		The purpose of this function is to get the statistics of accuracies of the model 
		onn ann unknown test set.
		we generate a confusion matrix and a probabibility matrix in the form of a heatmap and also 
		a csv file.
		"""

		self.confusion_matrix = np.zeros((self.n_classes,self.n_classes))
		self.probability_matrix = np.zeros((self.n_classes,self.n_classes))
	
		def confusion_matrix(results,GT):
			GT=np.argmax(GT,axis=1)
			result_one_hot=np.argmax(results,axis=1)
			(test_size,)=GT.shape
			cumsum=np.bincount(GT)+0.001

			for i in range(test_size):
				self.confusion_matrix[GT[i],:]+=results[i]/cumsum[GT[i]]	
				self.probability_matrix[GT[i],result_one_hot[i]]+=1.0/cumsum[GT[i]]
			
		confusion_matrix(results,GT)
		confusion_heat = plt.imshow(self.confusion_matrix, cmap='hot')
		plt.colorbar(confusion_heat, orientation='horizontal')
		plt.savefig(str(self.result_dir+'/'+'confusion_matrix.png'))
		plt.close()
		
		probability_heat = plt.imshow(self.probability_matrix, cmap='hot')
		plt.colorbar(probability_heat, orientation='horizontal')
		plt.savefig(str(self.result_dir+'/'+'probability_matrix'))
		plt.close()

	
		np.savetxt(self.result_dir+'/'"confusion_matrix.csv", self.confusion_matrix, delimiter=",")
		np.savetxt(self.result_dir+'/'"probability_matrix.csv", self.probability_matrix,delimiter=",")
		


	def plot_all(self):
		"""
		This function is called to plot the logs of training the neural net
		the plots generated are training loss , validation loss,train acc annd validationn acc  
		"""
		plt.plot(self.run_model.history['loss'])
		plt.plot(self.run_model.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.result_dir+'/'+'model_loss.png')

		plt.plot(self.run_model.history['acc'])
		plt.plot(self.run_model.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.result_dir+'/'+'model_accuracy.png')	
	


	def test(self,conf = False):
		X_test,Y_test = self.images_and_labels("testing")
		print(X_test.shape)
		results = self.model.predict(X_test)
		crap=self.model.predict(preprocess_input(np.zeros((10,200,200,3))))
		print(crap)
		print(type(X_test))
		self.get_statistics(results,Y_test)
			
	def callback(self,img):
		"""
		this function under the assumption that it receives a 4-D (number of images,width,height,1) sized image
		and use the predict method of the model class 
		the result generated is a (nnumber of images,number of classes) sized probability distribution for
		wach image sample
		"""
		try:
			ros_img = bridge.imgmsg_to_cv2(img, "bgr8")
			# ros_img = ros_numpy.numpify(img)
			# im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
			# print(im)
			#assert str(type(ros_img)) == "<type 'numpy.ndarray'>"
			if ros_img is not None:
				H,W,C=ros_img.shape
				feed_list=[]
				for i in range(W//self.img_cols):
					"""
					Below mentioned is the equation for the greyscale conversion algorithms as used by the PIL module
					I have kept it the same as to not allow conflict.
					The images are read snippets of 200,200,3 col wise and concerted to greyscale images (200,200,1) which is duplicated along the channel axis to make the 
					input 200,200,3 as this is the shape of the input the Neural net takes
					"""
					cv2.imshow("sign",ros_img)
					cv2.waitKey(1)

					ros_img=ros_img.astype("float64")
					GreyScaleImg=0.114*ros_img[:,i*200:i*200+200,0]+0.587*ros_img[:,i*200:i*200+200,1]+0.299*ros_img[:,i*200:i*200+200,2]
					feed_list.append([GreyScaleImg,GreyScaleImg,GreyScaleImg])

				FeedArray=np.array(feed_list,dtype=np.float64).transpose(0,2,3,1)	
				result_vector=self.model.predict(preprocess_input(FeedArray))
				PredictedIndex=np.argmax(result_vector,axis=1)
				print(PredictedIndex)
				
		except CvBridgeError as e:
			print('this is bad')
			pass

						

	def ros(self):
		#this function is the listener 
		print('ros successfully called')
		rospy.init_node('image_listener', anonymous=True)
		rospy.Subscriber('/trafficsignimage',Image,self.callback)
		#print(Image, "hahaha")
		rospy.spin()
			


parser = ArgumentParser()
parser.add_argument("-m", "--mode", dest="action",default='train')

args = parser.parse_args()


if(args.action=='train'):
	net = Resnet("training")
	net.train()

elif(args.action=='test'):
	net = Resnet("testing")
	net.test()

elif(args.action=='ros'):
	net = Resnet("ros")
	net.ros()
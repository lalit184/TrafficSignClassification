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
"""
from darknet_ros.msg import Array_images
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import rospy

bridge = CvBridge()
"""
class Resnet(params):
	def __init__(self,condition):
		super(Resnet, self).__init__()
		if condition == "training":
			self.model = Classifier(self.input_tensor,self.n_classes)
			
			for layer in self.model.layers:
				layer.trainable = True
		else:
			self.model = load_model('./check_point/weights-04-0.05.h5')

			
		
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
		results = self.model.predict(X_test)
		self.get_statistics(results,Y_test)
	"""		
	def callback(self, Array_images):
		
		this function under the assumption that it receives a 4-D (number of images,width,height,1) sized image
		and use the predict method of the model class 
		the result generated is a (nnumber of images,number of classes) sized probability distribution for
		wach image sample

		
		try:

			result_vector = self.model.predict(Array_images)
			predictedIndex,predictionProbability = np.argmax(result_vector, axis=1),np.amax(result_vector,axis=1)
			for i in range(predictedIndex.shape[0]):
				print(self.ClassesID[predictedIndex],"  and the probability is:",predictionProbability)

			print("------------------------------------------------------")
						
		except CvBridgeError as e:
			print('error came')

						

	def ros(self):
		#this function is the listener 
		self.data = data = np.zeros((1,self.img_rows,self.img_cols,self.img_channels))
		rospy.init_node('image_listener', anonymous=True)
		rospy.Subscriber('/trafficsignimage',Array_images,self.callback) #4D image successfuly received
		self.detected = 0
		rospy.spin()
			
	"""

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
import glob
import cv2

for i in range(28):
	AnnotationFileList=[]

	for file in glob.glob("./Data/"+str(i)+"/"+"*.png"):
		AnnotationFileList.append(file)
		#print(file)
	for file in AnnotationFileList:
		im=cv2.imread(file)
		print(file)	
		im=cv2.resize(im,(200,200))
		cv2.imwrite(file,im)

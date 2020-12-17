try:
	# for Python2
	import Tkinter as tk
	from Tkinter import filedialog
	import Queue
except ImportError:
	# for Python3
	import tkinter as tk
	from tkinter import filedialog
	import queue

import numpy as np
import cv2
import glob
import natsort
from PIL import Image, ImageTk
import threading
import argparse
import sys
import os
import time
import math
import yaml

from pdb import set_trace as bp

"""
for ip camera
"""
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

camerapath = ''

processframes  = queue.Queue()

class MainGUI:

	def __init__(self, vs, args, savedir):
		print ('----- Initialize -----')
		self.frame = None
		self.vs    = vs
		self.args  = args
		self.savedir = savedir
		self.end = False

		self.framenumber    =0
		self.finishprocess  =False
		
		self.parameterdir = self.savedir+os.sep+'parameter'
		os.makedirs(self.parameterdir, exist_ok=True)
		self.resultdir = self.savedir+os.sep+'resultimage'
		os.makedirs(self.resultdir, exist_ok=True)
		if args.sourcetype == 'image':
			files = glob.glob(self.vs + os.sep + '*.' + args.extension)
			self.imagefiles = natsort.natsorted(files,reverse=False)
			self.fps = 1
		elif args.sourcetype == 'video':
			self.vs = cv2.VideoCapture(self.vs)
			self.fps = 30
		"""
		termination criteria
		"""
		#self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)

		"""
		prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		quare_size = 85x88 mm
		"""
		numq = self.args.grid_number_inner_corners.split('x')
		self.columnq = int(numq[0])
		self.rowq    = int(numq[1])
		self.quare_size = [self.columnq, self.rowq]

		mmq = self.args.grid_square_size.split('x')
		self.colmm = float(mmq[0])
		self.rowmm = float(mmq[1])
		self.quare_mm_size = np.array([self.rowmm,self.colmm,0], np.float32)
		self.objp       = np.zeros((self.columnq*self.rowq,3), np.float32)
		self.objp[:,:2] = np.mgrid[0:self.rowq,0:self.columnq].T.reshape(-1,2)
		self.objp       = np.multiply(self.objp, self.quare_mm_size)
		#self.objp       = np.reshape(self.objp,(self.objp.shape[0],1,self.objp.shape[1]))


		# imagesize = self.args.image_size.split('x')
		# self.imagew = int(imagesize[0])
		# self.imageh = int(imagesize[1])
		# self.image_size = [self.imagew, self.imageh]

		"""
		Arrays to store object points and image points from all the images.
		"""
		self.objpoints = [] # 3d points in real world space.
		self.imgpoints = [] # 2d points in image plane.

		"""
		GUI show frames
		"""
		self.root   = tk.Tk()
		image_label = tk.Label(master=self.root)
		image_label.pack(side="left", padx=10, pady=10)

		self.stopEvent = threading.Event()
		self.stopEvent.set()

		self.readthread    = threading.Thread(target=self.ReadFrames, args=(image_label,) )
		self.processthread = threading.Thread(target=self.ProcessFrames, args=(image_label,) )
		self.readthread.start()
		self.processthread.start()

		self.root.wm_title("IVCLabCameraCalib")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

		print ('Initialize Successfully ...')

	def onClose(self):

		print ("----Calculate Camera Intrinsic Parameter----")
		starttime = time.time()
		frame_dims = (self.image_size[0],self.image_size[1])

		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=self.objpoints, 
			imagePoints=self.imgpoints, imageSize=frame_dims, cameraMatrix=None, distCoeffs=None)
		self.WriteResult(mtx, dist)
	
		total_error = 0
		for i in range(len(self.objpoints)):
			imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(self.imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			total_error += error
		print ("Camera Intrinsic Parameter: ")
		print (mtx)
		print ("distortion coefficients (k_1, k_2, p_1, p_2,k_3,k_4,k_5,k_6): ")
		print (dist)
		print ("mean error: ", total_error/len(self.objpoints))
		endtime = time.time()
		spendtime = endtime-starttime
		print ("spend time: %f" %(spendtime) )
		print ("onClose")
		print ("Closing .... ")
		with processframes.mutex:
			processframes.queue.clear()

		self.stopEvent.clear()
		if not isinstance(self.vs, str):
			self.vs.release()
		self.root.destroy()
		

	def ReadFrames(self, image_label):
		print ("----- Read Frames -----")
		try:
			frame_counter = 0
			while self.stopEvent.is_set():

				if self.args.sourcetype == 'image':
					if frame_counter < len(self.imagefiles):
						self.frame = cv2.imread(self.imagefiles[frame_counter])
					else:
						self.end = True
						print("----- Images End-----")
						break
					
				elif self.args.sourcetype == 'video':				
					ret_val, self.frame = self.vs.read()
					if self.frame is None:
						self.end = True
						print("----- Video End-----")
						break
				# resize to small images
				self.imagew = self.frame.shape[1]
				self.imageh = self.frame.shape[0]
				self.image_size = [self.imagew, self.imageh]
				self.frame = cv2.resize(self.frame, (self.imagew, self.imageh))
				image = self.frame
				if frame_counter % self.fps == 0:
					processframes.put(image)
				frame_counter += 1

		except RuntimeError as e:
			print ("caught a RuntimeError")

	def ProcessFrames(self, image_label):
		print ("----- Process Frames -----")
		try:
			while self.stopEvent.is_set():
				if (not processframes.empty() ):
					image = processframes.get()
					grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

					# Find the chess board corners
					ret, corners = cv2.findChessboardCorners(grayimage, (self.rowq,self.columnq),None)

					# If found, add object points, image points (after refining them)
					if ret == True:
						self.objpoints.append(self.objp)
						# default
						# winsize = 11
						winsize = max(5, int(round(cv2.norm(corners[0]-corners[1],cv2.NORM_INF)/4)) )
						corners2 = cv2.cornerSubPix(grayimage,corners,(winsize,winsize),(-1,-1),self.criteria)

						# reorder corner points
						image,corners2 = self.ReorderPoints(corners2, image)
						self.imgpoints.append(corners2)

						# Draw the corners
						image = cv2.drawChessboardCorners(image, (self.rowq,self.columnq), corners2,ret)
						cv2.imwrite(self.resultdir+os.sep+str(self.framenumber)+'.jpg',image)
						self.framenumber = self.framenumber+1


					self.finishprocess = True
					if self.finishprocess is True:
						processimage = Image.fromarray(image)
						processimage = ImageTk.PhotoImage(image=processimage)
						image_label.configure(image=processimage)
						image_label._image_cache = processimage
						self.finishprocess = False
				elif self.end:
					self.onClose()

		except RuntimeError as e:
			print ("caught a RuntimeError")

	def positiveIntRound(self,value):
		return int(value+0.5)

	def invertXPositionsIndices(self, cornerpoints, squaresize):

		# invert points orientation within each row
		for x in range(int(squaresize[1]/2)):
			for y in range(squaresize[0]):
				rowComponent = squaresize[1]*y
				temp = cornerpoints[rowComponent+x].copy()
				temp1= cornerpoints[rowComponent+(squaresize[1]-1)-x].copy()
				cornerpoints[rowComponent+x] = temp1
				cornerpoints[rowComponent+(squaresize[1]-1)-x] = temp

		return cornerpoints

	def ReorderPoints(self, cornerpoints, image):

		outterCornerIndices = [0, self.quare_size[0]-1, cornerpoints.shape[0]-self.quare_size[0], cornerpoints.shape[0]-1]
		fourPointsVector = [cornerpoints[outterCornerIndices[0]], cornerpoints[outterCornerIndices[1]],
							cornerpoints[outterCornerIndices[2]], cornerpoints[outterCornerIndices[3]]]
		point01     = fourPointsVector[0]-fourPointsVector[1]
		point01Norm = cv2.norm(point01)
		point02     = fourPointsVector[0]-fourPointsVector[2]
		point02Norm = cv2.norm(point02)
		point13     = fourPointsVector[1]-fourPointsVector[3]
		point13Norm = cv2.norm(point13)
		point32     = fourPointsVector[3]-fourPointsVector[2]
		point32Norm = cv2.norm(point32)
		averageSquareSizePx = (point01Norm/self.quare_size[0]+point02Norm/self.quare_size[1]+point13Norm/self.quare_size[1]+point32Norm/self.quare_size[0])/4

		# how many pixels does the outter square has?
		# 0.67 is a threshold to be safe
		diagonalLength = 0.67*math.sqrt(2)*averageSquareSizePx

		# In which direction do I have to look?
		# Normal vector between corners 0-1, 0-2, 1-3?
		point01Direction = 1 / point01Norm*point01
		point02Direction = 1 / point02Norm*point02
		point13Direction = 1 / point13Norm*point13

		# Initialization
		pointDirection = []
		pointDirection.append(1/cv2.norm(point01Direction+point02Direction)*(point01Direction+point02Direction))
		temppd = pointDirection[0].copy()
		temppd[0][1] = temppd[0][1]*-1 
		pointDirection.append(temppd)
		temppd = pointDirection[0].copy()
		temppd[0][0] = temppd[0][0]*-1
		pointDirection.append(temppd)
		pointDirection.append(-pointDirection[0])

		# Get line to check whether outter grid color is black
		pointLimit = []
		for i in range(len(fourPointsVector)):
			pointLimit.append(fourPointsVector[i]+diagonalLength*pointDirection[i])

		# Line search to see if white or black
		meanPxValues=[]
		numberPointsInLine = 25
		for i in range(len(fourPointsVector)):
			summ  = 0
			count = 0
			for j in range(numberPointsInLine):
				mX = max(0, min(self.image_size[0]-1, self.positiveIntRound(fourPointsVector[i][0][0]+j*pointDirection[i][0][0]) ) )
				mY = max(0, min(self.image_size[1]-1, self.positiveIntRound(fourPointsVector[i][0][1]+j*pointDirection[i][0][1]) ) )
				bgrValue = image[mY, mX]
				summ = summ+(float(bgrValue[0])+float(bgrValue[1])+float(bgrValue[2]))/3
				count = count+1
			meanPxValues.append(summ/count)

		# Get black indexes
		blackIs0 = meanPxValues[0] < meanPxValues[3]
		blackIs1 = meanPxValues[1] < meanPxValues[2]
		
		#print ("blackIs0: ")
		#print (blackIs0)
		#print ("blackIs1: ")
		#print (blackIs1)

		# Apply transformations
		# For simplicity, we assume 0 is black
		if (not blackIs0):
			cornerpoints = cornerpoints[::-1]
			blackIs1 = not blackIs1

		#for i in range(len(fourPointsVector)):
		#	cv2.line(image, (fourPointsVector[i][0][0],fourPointsVector[i][0][1]), (pointLimit[i][0][0],pointLimit[i][0][1]), (0,0,255), 10)

		# Lead is 0 or 1||2 (depending on blackIs1)?
		outterCornerIndicesAfter = [0, self.quare_size[0]-1, cornerpoints.shape[0]-self.quare_size[0], cornerpoints.shape[0]-1]
		middle = 0.25*(fourPointsVector[0] + fourPointsVector[1] + fourPointsVector[2] + fourPointsVector[3])
		fourPointsVectorAfter = [cornerpoints[outterCornerIndicesAfter[0]]-middle, cornerpoints[outterCornerIndicesAfter[1]]-middle,
								cornerpoints[outterCornerIndicesAfter[2]]-middle, cornerpoints[outterCornerIndicesAfter[3]]-middle]

		if (blackIs1):
			crossProduct = np.cross(fourPointsVectorAfter[0],fourPointsVectorAfter[1])
		else:
			crossProduct = np.cross(fourPointsVectorAfter[0],fourPointsVectorAfter[2])

		leadIs0 = crossProduct < 0
		# second transformation
		if (not leadIs0):
			# second black is 1
			if (blackIs1):
				# 1-> 0
				cornerpoints = self.invertXPositionsIndices(cornerpoints, self.quare_size)
			else: # second black is 2
				#2->3
				cornerpoints = cornerpoints[::-1]
				#3->0
				cornerpoints = self.invertXPositionsIndices(cornerpoints, self.quare_size)

		return image, cornerpoints
	
	def WriteResult(self, mtx, dist):
		with open(self.parameterdir+os.sep+'Config.yaml', 'w') as file:
			file.write('%YAML:1.0\n')
			result = {
				'scale': float(0.5),
				# Camera calibration and distortion parameters (OpenCV) 
				'Camera.fx': float(mtx[0, 0]),
				'Camera.fy': float(mtx[1, 1]),
				'Camera.cx': float(mtx[0, 2]),
				'Camera.cy': float(mtx[1, 2]),

				'Camera.k1': float(dist[0, 0]),
				'Camera.k2': float(dist[0, 1]),
				'Camera.p1': float(dist[0, 2]),
				'Camera.p2': float(dist[0, 3]),
				'Camera.k3': float(dist[0, 4]),

				# Camera frames per second 
				'Camera.fps': 30.0,

				# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
				'Camera.RGB': 1,

				#--------------------------------------------------------------------------------------------
				# ORB Parameters
				#--------------------------------------------------------------------------------------------

				# ORB Extractor: Number of features per image
				'ORBextractor.nFeatures': 2000,

				# ORB Extractor: Scale factor between levels in the scale pyramid 	
				'ORBextractor.scaleFactor': 1.2,

				# ORB Extractor: Number of levels in the scale pyramid	
				'ORBextractor.nLevels': 8,

				# ORB Extractor: Fast threshold
				# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
				# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
				# You can lower these values if your images have low contrast			
				'ORBextractor.iniThFAST': 20,
				'ORBextractor.minThFAST': 7,

				#--------------------------------------------------------------------------------------------
				# Viewer Parameters
				#--------------------------------------------------------------------------------------------
				'Viewer.KeyFrameSize': 0.05,
				'Viewer.KeyFrameLineWidth': 1,
				'Viewer.GraphLineWidth': 0.9,
				'Viewer.PointSize': 2,
				'Viewer.CameraSize': 0.08,
				'Viewer.CameraLineWidth': 3,
				'Viewer.ViewpointX': 0,
				'Viewer.ViewpointY': -0.7,
				'Viewer.ViewpointZ': -1.8,
				'Viewer.ViewpointF': 500
			}
			documents = yaml.dump(result, file)

def main(args):
	cam = None
	if args.sourcetype == 'image':
		path = args.filepath
		savedir = args.savepath
		try:
			os.mkdir(savedir)
		except OSError:
			print ("Creation of the directory %s failed" % savedir)
		else:
			print ("Successfully created the directory %s " % savedir)

	elif args.sourcetype =='video':
		path = args.filepath
		savedir = args.savepath
		try:
			os.mkdir(savedir)
		except OSError:
			print ("Creation of the directory %s failed" % savedir)
		else:
			print ("Successfully created the directory %s " % savedir)

	time.sleep(2.0)
	maingui = MainGUI(path, args, savedir)
	maingui.root.mainloop()

	print (" Exit. please close the terminal")

def parse_arguments(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('--sourcetype',type=str, default='image',
		help='sourcetype: image or video')
	parser.add_argument('--filepath', type=str, default='images',
		help='images or video path for calibartion')
	parser.add_argument('--savepath', type=str, default='intrinsics',
		help='folder that save calibrated image and intrinsic parameter')
	parser.add_argument('--grid-square-size', type=str, default="85x88",
		help='grid square size of printed grid (mm)')
	parser.add_argument('--grid-number-inner-corners', type=str, default="6x7",
		help='detect inner corners of printed chess, column x row')
	parser.add_argument('--extension', type=str, default='jpg',
		help='extension: jpg, png, mp4, avi, MOV ...etc')

	return parser.parse_args(argv)


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
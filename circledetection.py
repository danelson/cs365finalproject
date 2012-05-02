import sys
import numpy
import cv, cv2
import pipeline
import scipy.ndimage.filters as filters
import source
import imgutil
import scipy.ndimage.morphology as morphology
import math


class Display(pipeline.ProcessObject):
	'''
	'''
	def __init__(self, inpt = None, name = "pipeline"):
		pipeline.ProcessObject.__init__(self, inpt)
		cv2.namedWindow(name, cv.CV_WINDOW_NORMAL)
		self.name = name
		
	def generateData(self):
		inpt = self.getInput(0).getData()
		
		# Convert back to OpenCV BGR from RGB
		if inpt.ndim == 3 and inpt.shape[2] == 3:
			inpt = inpt[..., ::-1]
		
		cv2.imshow(self.name, inpt.astype(numpy.uint8))


class Grayscale(pipeline.ProcessObject):
	'''
	Converts an image to grayscale if it has 3 channels
	'''
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt)
	
	def generateData(self):
		inpt = self.getInput(0).getData()
	
		if inpt.ndim == 3 and inpt.shape[2]==3:
			output = inpt[...,0]*0.114 + inpt[...,1]*0.587 + inpt[...,2]*0.229
	
		output = output.astype(numpy.uint8)
		self.getOutput(0).setData(output)

class DrawCircles(pipeline.ProcessObject):
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt, inputCount=2)

	def generateData(self):
		inpt = self.getInput(0).getData().copy()
		circles = self.getInput(1).getData()
		
		for r, y, x in circles:
			cv2.circle(inpt, (x,y), r, (0,0,255), 2)
		self.getOutput(0).setData(inpt)
	
class CVCircles(pipeline.ProcessObject):
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt)
		self.minDist = 45 #10
		self.dp = 2.74 #1.5
	
	def generateData(self):
		inpt = self.getInput(0).getData()
		circles = cv2.HoughCircles(inpt, cv.CV_HOUGH_GRADIENT,self.dp, self.minDist)
		if circles is None:
			circles = [[(0,0,0)]]
		self.getOutput(0).setData(numpy.array(circles)[0,:,::-1])
		
	def setDP(self, dp):
		self.dp=dp
		self.modified()
	
	def setMinDist(self, minDist):
		self.minDist = minDist
		self.modified()

class HoughCircles(pipeline.ProcessObject):
	def __init__(self, inpt = None):
		pipeline.ProcessObject.__init__(self, inpt, outputCount=2)
		self.sigmaD = 1.0
		self.max_radius = 40
		self.min_radius = 10
		self.gradient_threshold = 50.0
		self.rad_idx = 20

	def generateData(self):
		orig_inpt = self.getInput(0).getData()
		
		
		inpt = orig_inpt.astype(numpy.float32)
		
		# compute derivatives in x
		Ix = filters.gaussian_filter1d(inpt, self.sigmaD, 0, 0)
		Ix = filters.gaussian_filter1d(Ix, self.sigmaD, 1, 1)
		
		# compute derivatives in y
		Iy = filters.gaussian_filter1d(inpt, self.sigmaD, 1, 0)
		Iy = filters.gaussian_filter1d(Iy, self.sigmaD, 0, 1)

		
		gradMag = imgutil.normalize(numpy.sqrt(Ix*Ix+Iy*Iy).astype(numpy.uint8))
		gradAng = numpy.arctan2(Iy,Ix)
		
		#gradMag = morphology.binary_opening(gradMag, structure=numpy.ones((3,3)), iterations=3)
		
		#output = (gradMag>self.gradient_threshold)#cv2.Canny(orig_inpt, 10, 100)
		#output = morphology.binary_dilation(output, structure=numpy.ones((3,3)), iterations=4)
		#output = morphology.binary_erosion(output, structure=numpy.ones((3,3)), iterations=4)
		
		# non-maximum supression
		#output2 = findEdges(gradMag, gradAng, self.gradient_threshold)
		
		output = cv2.Canny(orig_inpt, 10, 100)
		
		#circles = cv2.HoughCircles(orig_inpt, cv.CV_HOUGH_GRADIENT,1.5, 10)[0]
		circles = []
		#if not self.max_radius:
		#	 self.max_radius = max(inpt.shape[0], inpt.shape[1])
		bins = numpy.zeros((self.max_radius, gradMag.shape[0], gradMag.shape[1]), numpy.uint8)
		print "Filling bins"
		for y,x in zip(*numpy.where(output)):
			r = self.min_radius
			theta0 = gradAng[y,x]
			theta = theta0-.5
			#TODO: vary theta by +- .5 or so
			while math.fabs(theta0+.5 - theta) > 1e-3:
				theta += .02			
				sinTheta = math.sin(theta)
				cosTheta = math.cos(theta)
				while r<self.max_radius-1:
					cY, cX = y+r*sinTheta, x+r*cosTheta 
					if cY>0 and cX>0 and cX<output.shape[1] and cY<output.shape[0]:
						bins[r, cY, cX]+= 1
					r += 1
			
		#TODO: find local maxima instead of global
		from scipy.ndimage.filters import maximum_filter
		print "bins filled"
		print bins.sum()		
		circles = zip(*numpy.where(bins > bins.max()-10))
		#circles = zip(*numpy.where(bins == maximum_filter(bins,20)))
		print "circles found",len(circles)

		
		self.getOutput(0).setData(circles)#bins[self.rad_idx]*100)
		self.getOutput(1).setData(bins[self.rad_idx]*100)
	
	def setSigmaD(self, sigmaD):
		self.sigmaD=sigmaD
		self.modified()
	
	def setGradientThreshold(self, gs):
		self.gradient_threshold = gs
		self.modified()
	
	def setRadIdx(self, idx):
		self.rad_idx = idx
		self.modified()


def findEdges(gradMag, gradAng, magThresh=200):
	'''
	Finds edges from gradient magnitude and orientation images.	 The orientation
	image is converted to a coarse approximation of angle, rounded to the
	nearest pi/4. 
	'''
	# make coarse version of gradient orientation--split into 4 pi/4 sections
	piOver8 = numpy.pi / 8
	coarseAng = numpy.copy(gradAng)
	coarseAng[coarseAng < 0] = coarseAng[coarseAng < 0] + numpy.pi
	coarseAng[numpy.logical_or( coarseAng < 1 * piOver8, coarseAng > 7 * piOver8)] = 0
	coarseAng[numpy.logical_and(coarseAng > 1 * piOver8, coarseAng < 3 * piOver8)] = 2 * piOver8
	coarseAng[numpy.logical_and(coarseAng > 3 * piOver8, coarseAng < 5 * piOver8)] = 4 * piOver8
	coarseAng[numpy.logical_and(coarseAng > 5 * piOver8, coarseAng < 7 * piOver8)] = 6 * piOver8

	# compute offsets for all pixels; 1.5 scaling ensures that diagonal 
	# directions will round to next pixel (cos(pi/4) = sqrt(2)/2)
	offsetRow = numpy.int8(1.5 * numpy.sin(coarseAng))
	offsetCol = numpy.int8(1.5 * numpy.cos(coarseAng))
		
	# threshold gradient magnitude to exclude weak edges, and exclude boundaries
	mag = numpy.where(gradMag >= magThresh, 1, 0) * gradMag
	mag[0,:] = 0
	mag[:,0] = 0
	mag[mag.shape[0]-1,:] = 0
	mag[:, mag.shape[1]-1] = 0
	(rIdx, cIdx) = numpy.where(mag > 0)
	
	# create edge image
	edge = numpy.zeros(mag.shape, numpy.uint8)
	
	# set as edges locations where gradient is high and a local maximum along
	# the gradient direction
	for r,c in zip(rIdx, cIdx):
		dr, dc = offsetRow[r,c], offsetCol[r,c]
		if mag[r, c] >= mag[r + dr, c + dc] and mag[r, c] >= mag[r - dr, c - dc]:
			edge[r,c] = 1
			
	return edge




def test_circle_detection(args):
	pipesource = source.FileReader("./data/2.png")
	grayScale = Grayscale(pipesource.getOutput())
	hCircles = HoughCircles(grayScale.getOutput())
	cvCircles = CVCircles(grayScale.getOutput())
	
	drawHCircles = DrawCircles()
	drawHCircles.setInput(grayScale.getOutput(0),0)
	drawHCircles.setInput(hCircles.getOutput(0), 1)
	
	drawCvCircles = DrawCircles()
	drawCvCircles.setInput(grayScale.getOutput(0),0)
	drawCvCircles.setInput(cvCircles.getOutput(0), 1)
	
	display0 = Display(pipesource.getOutput(0), "")
	display = Display(drawHCircles.getOutput(0), "Our Hough")
	
	display2 = Display(drawCvCircles.getOutput(0), "Their Hough")

	key = None
	while key != 27:
		display0.update()
		display.update()
		
		display2.update()
		key = cv2.waitKey(10)
		
		if key == 106: #j
			cvCircles.setDP(cvCircles.dp+0.02)
			print cvCircles.dp
		if key == 108: #l
			cvCircles.setDP(cvCircles.dp-0.02)
			print cvCircles.dp
		if key == 105: #i
			cvCircles.setMinDist(cvCircles.minDist+5)
			print cvCircles.minDist
		if key == 107: #k
			cvCircles.setMinDist(cvCircles.minDist-5)
			print cvCircles.minDist
		
		if key == 119: #w
			hCircles.setRadIdx(hCircles.rad_idx+1)
			print hCircles.rad_idx
			#hCircles.setSigmaD(hCircles.sigmaD+0.02)
			#print hCircles.sigmaD
		if key == 115: #s
			hCircles.setRadIdx(hCircles.rad_idx-1)
			print hCircles.rad_idx
			#hCircles.setSigmaD(hCircles.sigmaD-0.02)
			#print hCircles.sigmaD
		if key == 97: #a
			hCircles.setGradientThreshold(hCircles.gradient_threshold+5)
			print hCircles.gradient_threshold
		if key == 100: #d
			hCircles.setGradientThreshold(hCircles.gradient_threshold-5)
			print hCircles.gradient_threshold
		
	
if __name__ == "__main__":
	test_circle_detection(sys.argv)
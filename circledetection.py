import sys
import numpy
import cv, cv2
import pipeline
import scipy.ndimage.filters as filters
import source
import imgutil


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


class HoughCircles(pipeline.ProcessObject):
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt)
        self.sigmaD = 1.0

    def generateData(self):
		inpt = self.getInput(0).getData()
		
		
		inpt = inpt.astype(numpy.float32)
		
		# compute derivatives in x
		Ix = filters.gaussian_filter1d(inpt, self.sigmaD, 0, 0)
		Ix = filters.gaussian_filter1d(Ix, self.sigmaD, 1, 1)
		
		# compute derivatives in y
		Iy = filters.gaussian_filter1d(inpt, self.sigmaD, 1, 0)
		Iy = filters.gaussian_filter1d(Iy, self.sigmaD, 0, 1)


		gradMag = imgutil.normalize(numpy.sqrt(Ix*Ix+Iy*Iy).astype(numpy.uint8))
		
		
		circles = cv2.HoughCircles(gradMag, cv.CV_HOUGH_GRADIENT,1, 80)[0]
		
		for x, y, r in circles:
			cv2.circle(inpt, (x,y), r, (0,0,255), 2)
		
		self.getOutput(0).setData(inpt)

def test_circle_detection(args):
    pipesource = source.FileReader("/Users/hnphan/desktop/cs365finalproject/data/1.png")
    grayScale = Grayscale(pipesource.getOutput())
    hCircles = HoughCircles(grayScale.getOutput())
    display = Display(hCircles.getOutput())

    key = None
    while key != 27:
        display.update()
        key = cv2.waitKey(100)
    
if __name__ == "__main__":
    test_circle_detection(sys.argv)
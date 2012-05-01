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


class HoughCircles(pipeline.ProcessObject):
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt)
        self.sigmaD = 1.0
        self.max_radius = 100
        self.min_radius = 10
        self.gradient_threshold = 50.0

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
        
        #gradMag = morphology.binary_opening(gradMag, structure=numpy.ones((3,3)), iterations=3)
        
        output = (gradMag>self.gradient_threshold)#cv2.Canny(orig_inpt, 10, 100)
        output = morphology.binary_dilation(output, structure=numpy.ones((3,3)), iterations=4)
        output = morphology.binary_erosion(output, structure=numpy.ones((3,3)), iterations=4)*255
        
        #circles = cv2.HoughCircles(orig_inpt, cv.CV_HOUGH_GRADIENT,1.5, 10)[0]
        circles = []
        #if not self.max_radius:
        #    self.max_radius = max(inpt.shape[0], inpt.shape[1])
        print gradMag.max(), gradMag.min(), gradMag.mean()
        bins = numpy.zeros((self.max_radius, gradMag.shape[0], gradMag.shape[1]), numpy.uint8)
        print output.size
        for y,x in zip(*numpy.where(output)):
            r = self.min_radius
            r = 20
            theta = math.atan2(Iy[y,x],Ix[y,x])
            print theta
            while r<self.max_radius-1:
                bins[r, r*math.sin(theta), r*math.cos(theta)]+= 1
                r+= self.max_radius
        
        
                
            
        
        for x, y, r in circles:
            cv2.circle(inpt, (x,y), r, (0,0,255), 2)
        
        self.getOutput(0).setData(bins[20])
    
    def setSigmaD(self, sigmaD):
        self.sigmaD=sigmaD
        self.modified()
    
    def setGradientThreshold(self, gs):
        self.gradient_threshold = gs
        self.modified()

def test_circle_detection(args):
    pipesource = source.FileReader("/Users/danelson/Desktop/cs365final/data/2.png")
    grayScale = Grayscale(pipesource.getOutput())
    hCircles = HoughCircles(grayScale.getOutput())
    display = Display(hCircles.getOutput())

    key = None
    while key != 27:
        display.update()
        key = cv2.waitKey(10)
        if key == 119:
            hCircles.setSigmaD(hCircles.sigmaD+0.02)
            print hCircles.sigmaD
        if key == 115:
            hCircles.setSigmaD(hCircles.sigmaD-0.02)
            print hCircles.sigmaD
        if key == 97:
            hCircles.setGradientThreshold(hCircles.gradient_threshold+5)
            print hCircles.gradient_threshold
        if key == 100:
            hCircles.setGradientThreshold(hCircles.gradient_threshold-5)
            print hCircles.gradient_threshold
        
    
if __name__ == "__main__":
    test_circle_detection(sys.argv)
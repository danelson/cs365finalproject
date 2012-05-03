import sys
import math

import numpy
import cv, cv2

import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

import pipeline
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


class Gaussian(pipeline.ProcessObject):
    '''
    Converts an image to grayscale if it has 3 channels
    '''
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt)
        self.sigma = 1.5
    
    def generateData(self):
        inpt = self.getInput(0).getData()
    
        output = filters.gaussian_filter(inpt,self.sigma)
        
        self.getOutput(0).setData(output)
        
    def setSigma(self, sigma):
        self.sigma = sigma
        self.modified()
        
        
class Edges(pipeline.ProcessObject):
    '''
    Finds edges of an image
    '''
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt)
    
    def generateData(self):
        inpt = self.getInput(0).getData()
    
        output = cv2.Canny(inpt, 10, 100)
        
        self.getOutput(0).setData(output)
        
        
class Gradient(pipeline.ProcessObject):
    '''
    Finds edges of an image
    '''
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt, outputCount=2)
        self.sigmaD = 1.0
    
    def generateData(self):
        inpt = self.getInput(0).getData().astype(numpy.float32)
    
        # compute derivatives in x
        Ix = filters.gaussian_filter1d(inpt, self.sigmaD, 0, 0)
        Ix = filters.gaussian_filter1d(Ix, self.sigmaD, 1, 1)
        
        # compute derivatives in y
        Iy = filters.gaussian_filter1d(inpt, self.sigmaD, 1, 0)
        Iy = filters.gaussian_filter1d(Iy, self.sigmaD, 0, 1)

        # compute gradient magnitude and angle
        gradMag = imgutil.normalize(numpy.sqrt(Ix*Ix+Iy*Iy).astype(numpy.uint8))
        gradAng = numpy.arctan2(Iy,Ix)
        
        self.getOutput(0).setData(gradMag)
        self.getOutput(1).setData(gradAng)


class DrawCircles(pipeline.ProcessObject):
    '''
    '''
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt, inputCount=2)

    def generateData(self):
        inpt = self.getInput(0).getData().copy()
        
        circles = self.getInput(1).getData()
        for r, y, x in circles:
            cv2.circle(inpt, (x,y), r, (0,0,255), 2)
            
        self.getOutput(0).setData(inpt)
    
    
class CVCircles(pipeline.ProcessObject):
    '''
    '''
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
    '''
    '''
    def __init__(self, inpt = None):
        pipeline.ProcessObject.__init__(self, inpt, inputCount=4, outputCount=2)
        self.sigmaD = 1.0
        self.max_radius = 40
        self.min_radius = 10
        self.gradient_threshold = 50.0
        self.rad_idx = 20

    def generateData(self):
        orig_inpt = self.getInput(0).getData()
        edge = self.getInput(1).getData()
        gradMag = self.getInput(2).getData()
        gradAng = self.getInput(3).getData()
        
        inpt = orig_inpt.astype(numpy.float32)
        
        bins = numpy.zeros((self.max_radius, gradMag.shape[0], gradMag.shape[1]), numpy.uint8)
        print "Filling bins"
        for y,x in zip(*numpy.where(edge)):
            r = self.min_radius
            theta = gradAng[y,x]
            #TODO: vary theta by +- .5 or so
            sinTheta = math.sin(theta)
            cosTheta = math.cos(theta)
            while r<self.max_radius-1:
                cY, cX = y+r*sinTheta, x+r*cosTheta 
                if cY>0 and cX>0 and cX<edge.shape[1] and cY<edge.shape[0]:
                    bins[r, cY, cX] += 1
                r += 1
        
        #TODO: find local maxima instead of global
        print "bins filled"
        print "Bin sum", bins.sum()     
        print "Bin max", bins.max()
        #circles = zip(*numpy.where(bins > bins.max()-20))
        #circles = zip(*numpy.where(bins == maximum_filter(bins,20)))
        circles = local_maxima(bins)
        print "Number of circles",len(circles)
        
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


# Brian's function
def findEdges(gradMag, gradAng, magThresh=200):
    '''
    Finds edges from gradient magnitude and orientation images.  The orientation
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
    

def local_maxima(image, epsilon=45, threshold=9.5):
    image_max = filters.maximum_filter(image, epsilon)
    #find the location of a maxima on a patch
    maxima = (image == image_max)
    
    #make sure there is enough variation over the patch to warrant calling this a max
    image_min = filters.minimum_filter(image, epsilon)
    diff = ((image_max - image_min) > threshold)
    maxima[diff == 0] = 0
    
    return zip(*numpy.where(maxima))


def test_circle_detection(args):
    """
    pipesource = source.FileReader("./data/2.png")#"./data/ring.jpg")
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
    
    """
    pipesource = source.FileReader("./data/2.png")
    
    ### pipeline with smoothing
    gaussian1 = Gaussian(pipesource.getOutput())
    grayscale1 = Grayscale(gaussian1.getOutput())
    edges1 = Edges(grayscale1.getOutput())
    grad1 = Gradient(grayscale1.getOutput())
    
    hCircles1 = HoughCircles(grayscale1.getOutput())
    hCircles1.setInput(edges1.getOutput(), 1)
    hCircles1.setInput(grad1.getOutput(0), 2)
    hCircles1.setInput(grad1.getOutput(1), 3)
    cvCircles1 = CVCircles(grayscale1.getOutput())
    
    ## drawing circles
    # ours
    drawHCircles1 = DrawCircles()
    drawHCircles1.setInput(grayscale1.getOutput(0), 0)
    drawHCircles1.setInput(hCircles1.getOutput(0), 1)
    #theirs
    drawCvCircles1 = DrawCircles()
    drawCvCircles1.setInput(grayscale1.getOutput(0), 0)
    drawCvCircles1.setInput(cvCircles1.getOutput(0), 1)
    
    ### pipeline without smoothing
    grayscale2 = Grayscale(pipesource.getOutput())
    edges2 = Edges(grayscale2.getOutput())
    grad2 = Gradient(grayscale2.getOutput())
    
    hCircles2 = HoughCircles(grayscale2.getOutput())
    hCircles2.setInput(edges2.getOutput(), 1)
    hCircles2.setInput(grad2.getOutput(0), 2)
    hCircles2.setInput(grad2.getOutput(1), 3)
    cvCircles2 = CVCircles(grayscale2.getOutput())
    
    ## drawing circles
    # ours
    drawHCircles2 = DrawCircles()
    drawHCircles2.setInput(grayscale2.getOutput(0),0)
    drawHCircles2.setInput(hCircles2.getOutput(0), 1)
    # theirs
    drawCvCircles2 = DrawCircles()
    drawCvCircles2.setInput(grayscale2.getOutput(0),0)
    drawCvCircles2.setInput(cvCircles2.getOutput(0), 1)
    
    ### displays
    display1 = Display(pipesource.getOutput(0), "Source")
    # smoothing
    display2 = Display(drawHCircles1.getOutput(0), "Our Hough Smooth")
    display3 = Display(drawCvCircles1.getOutput(0), "Their Hough Smooth")
    # no smoothing
    display4 = Display(drawHCircles2.getOutput(0), "Our Hough Not Smooth")
    display5 = Display(drawCvCircles2.getOutput(0), "Their Hough Not Smooth")
    
    display6 = Display(edges1.getOutput(0), "Edges Smooth")
    display7 = Display(edges1.getOutput(0), "Edges Not Smooth")
    display8 = Display(gaussian1.getOutput(0), "Gaussian Smooth")
    
    
    key = None
    while key != 27:
        display1.update()
        display2.update()
        display3.update()
        display4.update()
        display5.update()
        display6.update()
        display7.update()
        display8.update()
        
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
        if key == 122: #z
            gaussian1.setSigma(gaussian1.sigma - 0.25)
            print gaussian1.sigma
        if key == 120: #x
            gaussian1.setSigma(gaussian1.sigma + 0.25)
            print gaussian1.sigma
            
    
if __name__ == "__main__":
    test_circle_detection(sys.argv)


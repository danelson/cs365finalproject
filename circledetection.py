import sys
import numpy
import cv, cv2

import source


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

def test_circle_detection(args):
    f = file("./data/258 5%% day 5 (20x).mov", 'r')
    pipesource = source.FileReader("./data/258 5%% day 5 (20x).mov")
    display = Dispaly(pipesource.getOutput())
    
    key = None
    while key != 27:
        display.update()
        key = cv2.waitkey(100)
    
if __name__ == "__main__":
    test_circle_detection(sys.argv)
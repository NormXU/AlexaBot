#!/usr/bin/env python

import cv2 as cv
import cv2
from common import anorm, clock
from functools import partial
import numpy as np

help_message = '''Good Features to Track

USAGE: good_features.py [ <image> ]
'''

if __name__ == '__main__':
    import sys
    try: fn1 = sys.argv[1]
    except:
        fn1 = "/home/norm/catkin_ws/src/rbx1/rbx1_vision/scripts/test_images/mona_lisa.png"
        
    print help_message
    
    # Good features parameters
    gf_params = dict( maxCorners = 200, 
                   qualityLevel = 0.1,
                   minDistance = 7,
                   blockSize = 20,
                   useHarrisDetector = False,
                   k = 0.04 )

    img = cv2.imread(fn1, cv2.IMREAD_COLOR)
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    start = clock()

    keypoints = cv2.goodFeaturesToTrack(grey, mask = None, **gf_params)
    print(keypoints)
    print(keypoints.reshape(-1,2))

    if keypoints is not None:
        for x, y in np.float32(keypoints).reshape(-1, 2):
            cv2.circle(img, (x, y), 3, (0, 255, 0, 0), 1, 8, 0)    

    
    print "Elapsed time:", 1000 * (clock() - start), "milliseconds"
    cv2.imshow("Keypoints", img)
    cv2.waitKey()

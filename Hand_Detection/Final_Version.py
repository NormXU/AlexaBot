
# coding: utf-8

# In[20]:


import cv2
import numpy as np
import math


# In[21]:


cap = cv2.VideoCapture(0)


# In[22]:


while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame,(170,170),(400,400),(0,255,0),0)
        crop_image = frame[170:400, 170:400]
        height, width, channels = crop_image.shape
        #print(height, width)
    
        blur = cv2.blur(crop_image,(3,3), 0)
    
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower_range = np.array([2,0,0])
        upper_range = np.array([16,255,255])
    
        mask = cv2.inRange(hsv,lower_range,upper_range)
    
        skinkernel = np.ones((5,5))
        #Kernel matrices for morphological transformation    
        dilation = cv2.dilate(mask, skinkernel, iterations = 1)
        erosion = cv2.erode(dilation, skinkernel, iterations = 1) 
    
        filtered = cv2.GaussianBlur(erosion, (15,15), 1)
        ret,thresh = cv2.threshold(filtered, 127, 255, 0)
    
        Label_ret, markers = cv2.connectedComponents(thresh)
        num = markers.max()
    
        #print(num)
        ## If the count of pixels less than a threshold, then set pixels to `0`.
        for i in range(1, num+1):
            pts =  np.where(markers == i)
            if len(pts[0]) < 150:
                markers[pts] = 0
    
        
        label_hue = np.uint8(markers.copy())
    
        label_hue = np.where(label_hue != 0, 255, label_hue)
        
    
    
    
        res = cv2.bitwise_and(crop_image,crop_image,mask = label_hue)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        
        
    #find contours
        _,contours,hierarchy= cv2.findContours(label_hue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90:
                l += 1
                #cv2.circle(crop_image, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(crop_image,start, end, [0,255,0], 2)
            
            
        #l+=1
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 4:    
            cv2.putText(frame,"Go Forward", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            
        #show the windows
        cv2.imshow('hand_binary',label_hue)
        cv2.imshow('frame',frame)
        cv2.imshow('hand',res)
    except:
        pass
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[23]:


cv2.destroyAllWindows()
cap.release()   


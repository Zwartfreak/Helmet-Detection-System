#!/usr/bin/env python
# coding: utf-8

# # Helmet detection

# **OpenCV-Python is a library of Python bindings designed to solve computer vision problems. 
# All the OpenCV array structures are converted to and from Numpy arrays. 
# This also makes it easier to integrate with other libraries that use Numpy such as SciPy and Matplotlib.**
# 

# In[1]:


# IMPORT NECESSARY LIBRARIES

import cv2 
import numpy as np 


# ### Classifier

# In[2]:


#HELMET CLASSIFIER
helmet_classifiers = cv2.CascadeClassifier(r'C:/Users/Shray/Anaconda3/Lib/site-packages/cv2/data/haarcascade_helmet.xml')


# #### In the below function, all the important part of the image is cropped and sent back as a result

# In[4]:


# FUNCTION 

def helmet_extract(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    helmets = helmet_classifiers.detectMultiScale(gray,1.3,5)
    
    if helmets is():  # In case of empty tuple/frame received
        return None
    
    for(x,y,w,h) in helmets:
        cropped_helmet = img[y:y+w, x:x+h]
        
    return cropped_helmet


# In[ ]:


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if helmet_extract(frame) is not None:
        count+=1
        helmet = cv2.resize(helmet_extract(frame),(200,200))
        helmet = cv2.cvtColor(helmet, cv2.COLOR_BGR2GRAY)
        
        file_name_path = 'S:/Project Final Year/Detected images/Helmet detected images/helmet'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,helmet)
        cv2.putText(helmet,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Helmet Croper',helmet)
    else:
        print('Helmet Not Detected')
        pass
     
    if cv2.waitKey(1)==13 or count==50:
        break
        
cap.release()
cv2.destroyAllWindows()

print('Collecting Samples Completed!!!!')
    


# In[6]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:





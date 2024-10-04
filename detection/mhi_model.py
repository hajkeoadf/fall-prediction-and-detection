import numpy as np
import cv2 
from time import sleep 
import queue
class MHI:
    def __init__(self,frame,tau,delta,xi,t):
        self.tau=tau
        self.delta=delta
        self.xi=xi
        self.t=t
        self.data = queue.Queue()
        for i in range(t):
            self.data.put(frame)
        self.H = np.zeros(frame.shape)    
    def getimag(self,frame):        
        self.data.put(frame)
        old_frame=self.data.get()        
        a=cv2.addWeighted(old_frame.astype(float),1, frame.astype(float), -1, 0)
        D= np.fabs(a)
        Psi= D >=self.xi        
        c=self.H-self.delta
        H=np.maximum(0,c)        
        H[Psi]=self.tau
        self.H=H
        return H.astype("uint8")


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import soundfile
import numpy 
import matplotlib.pyplot as pyplot
import IPython.display as diplay
import pyworld
import seaborn
import math
import sys


# In[2]:


import subprocess
subprocess.run(['jupyter','nbconvert','--to','python','func_mcep.ipynb'])


# In[8]:


def modify(data1,data,ran):
    newdata = numpy.array([])
    newdata = numpy.full([len(data),],data)
    midran = int(ran/2)
    for k in range(midran):
        newdata[len(data1)-midran+k] = fi(newdata[len(data1)-midran],newdata[len(data1)+midran],ran,k)
        newdata[len(data1)+k] = fi(newdata[len(data1)-midran],newdata[len(data1)+midran],ran,midran+k)
    return newdata


# In[9]:


def fi(ledata,ridata,size,x):
    y = ((ridata-ledata)/size*x) + ledata
    return y


# In[15]:


def hokan1(f01,f0,mcep1,mcep,ap1,ap,ran):
    newf0 = modify(f01,f0,ran)
    newmcep = numpy.full([mcep.shape[0],mcep.shape[1]],mcep)
    newap = numpy.full([ap.shape[0],ap.shape[1]],ap)
    for k in range(mcep.shape[1]):
        newmcep[:,k] = modify(mcep1[:,k],mcep[:,k],ran)
    for k in range(ap.shape[1]):
        newap[:,k] = modify(ap1[:,k],ap[:,k],ran)
    return newf0,newmcep,newap


# In[2]:


def connect(f01,f02,mcep1,mcep2,ap1,ap2): #補間せず接続
    f0 = numpy.hstack((f01,f02))
    mcep = numpy.vstack((mcep1,mcep2))
    ap = numpy.vstack((ap1,ap2))
    return f0,mcep,ap


# In[1]:


def makePitch(f0, pitch):#constant F0
    targetF0 = 440 * pow(2, (pitch - 69) / 12)
    newF0 = numpy.vectorize(lambda x: 0 if x == 0 else targetF0)(f0)
    return newF0.astype(numpy.double)


# In[4]:


def linear(data1,data2,point):#関数leng内で使用
    data = (data2 - data1) * point + data1
    if(point < 0 or point > 1):
        sys.stderr.write("please input 0 < point < 1\n")
    return data


# In[1]:


def leng(data,f0,mcep,ap,time,fs):
    oritime = len(data)/fs
    x = oritime / time
    
    newf0 = numpy.zeros(math.floor((len(f0) - 1.0) / x))
    newmcep = numpy.zeros((math.floor((mcep.shape[0] - 1.0) / x), mcep.shape[1]))
    newap = numpy.zeros((math.floor((ap.shape[0] - 1.0) / x), ap.shape[1]))
    
    ind = 0
    
    for k in range(len(newf0)):
        f,d = math.modf(k * x)
        
        for l in range(mcep.shape[1]):
            newmcep[k,l] = linear(mcep[math.floor(ind),l], mcep[math.ceil(ind),l], f)
        for l in range(ap.shape[1]):
            newap[k,l] = linear(ap[math.floor(ind),l], ap[math.ceil(ind),l], f)
            
        newf0[k] = linear(f0[math.floor(ind)], f0[math.ceil(ind)], f)
        
        ind = ind + x
        
    return newf0,newmcep,newap


# In[7]:


def hokan2(f01,f02,mcep1,mcep2,ap1,ap2,frame):
    ref01 = numpy.zeros(len(f01)-frame)
    remcep1 = numpy.zeros((mcep1.shape[0]-frame,mcep1.shape[1]))
    reap1 = numpy.zeros((ap1.shape[0]-frame,ap1.shape[1]))
    
    for k in range(len(ref01)):
        ref01[k] = f01[k]
        remcep1[k,:] = mcep1[k,:]
        reape1[k,:] = ap1[k,:]
        
    f0_hokan = numpy.zeros(frame)
    mcep_hokan = numpy.zeros((frame,mcep1,shape[1]))
    ap_hokan = numpy.zeros((frame,ap1,shape[1]))
    
    for k in range(frame):
        f0_hokan[k] = fi(ref01[len(ref01)-1],f02[int(len(f02)/2)],frame,k)
        for l in range(mcep1.shape[1]):
            mcep_hokan[k,l] = fi(remcep1[remcep1.shape[0]-1,l],mcep2[int(mcep2.shape[0]/2),l],frame,k)
        for l in range(ap1.shape[1]):
            ap_hokan[k,l] = fi(reap1[reap1.shape[0]-1,l],ap2[int(ap2.shape[0]/2),l],frame,k)
            
    newf0 = numpy.hstack((ref01,f0_hokan))
    newf0 = numpy.hstack((newf0,f02))
    newmcep = numpy.vstack((remcep1,mcep_hokan))
    newmcep = numpy.vstack((newmcep,mcep2))
    newap = numpy.vstack((reap1,ap_hokan))
    newap = numpy.vstack((newap,ap2))
    
    return newf0,newmcep,newap


# In[ ]:


def fall(f01,f02,fallframe,hokanframe,hnum):
    prep = numpy.zeros(fallframe)
    under = numpy.zeros(fallframe)
    inter = 180 / (fallframe - 1)
    rad = 0
    for k in range(len(prep)):
        prep[k] = 8 * math.sin(math.radians(rad))
        under[k] = -1.5 * prep[k]
        rad += inter
    
    newf01 = f01.copy()
    newf02 = f02.copy()
    
    if(hnum == 1):
        for k in range(fallframe):
            newf01[len(newf01)-int(hokanframe/2)-k] += prep[k]
            newf02[int(hokanframe/2)+k] += under[k]
    elif(hnum == 2):
        for k in range(fallframe):
            newf01[len(newf01)-hokanframe-k] += prep[k]
            newf02[k] += under[k]
    elif(hnum == 0):
        for k in range(fallframe):
            newf01[len(newf01)-k] += prep[k]
            newf02[k] += under[k]
    
    return newf01,newf02


# In[ ]:


def shakuri(f01,f02,fallframe,hokanframe,hnum):
    prep = numpy.zeros(fallframe)
    over = numpy.zeros(fallframe)
    inter = 180 / (fallframe - 1)
    rad = 0
    for k in range(len(prep)):
        prep[k] = -10 * math.sin(math.radians(rad))
        over[k] = -0.8 * prep[k]
        rad += inter
        
    newf01 = f01.copy()
    newf02 = f02.copy()
    
    if(hnum == 1):
        for k in range(fallframe):
            newf01[len(newf01)-int(hokanframe/2)-k] += prep[k]
            newf02[int(hokanframe/2)+k] += over[k]
    elif(hnum == 2):
        for k in range(fallframe):
            newf01[len(newf01)-hokanframe-k] += prep[k]
            newf02[k] += over[k]
    elif(hnum == 0):
        for k in range(fallframe):
            newf01[len(newf01)-k-1] += prep[k]
            newf02[k] += over[k]
        
    return newf01,newf02


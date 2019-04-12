#!/usr/bin/env python
# coding: utf-8

# In[7]:


import soundfile
import numpy as np
import matplotlib.pyplot as pyplot
import IPython.display as display
import pyworld as pw
import seaborn
import math
import sys
import pysptk as ps


# In[2]:


import subprocess
subprocess.run(['jupyter','nbconvert','--to','python','functions.ipynb'])


# In[20]:


def dataExport(file):
    dir = './RITSU/syllables/'
    data,fs = soundfile.read(dir + file)
    return data,fs


# In[1]:


def signalSeparation(data,fs):
    _f0,t = pw.dio(data,fs)
    f0 = pw.stonemask(data,_f0,t,fs)
    env = pw.cheaptrick(data,f0,t,fs)
    ape = pw.d4c(data,f0,t,fs)
    return f0,env,ape


# In[6]:


def fi(ledata,ridata,size,x):
    y = ((ridata-ledata)/size*x) + ledata
    return y


# In[7]:


def modify(data1,data,ran):
    newdata = np.array([])
    newdata = np.full([len(data),],data)
    midran = int(ran/2)
    for k in range(midran):
        newdata[len(data1)-midran+k] = fi(newdata[len(data1)-midran],newdata[len(data1)+midran],ran,k)
        newdata[len(data1)+k] = fi(newdata[len(data1)-midran],newdata[len(data1)+midran],ran,midran+k)
    return newdata


# In[8]:


def hokan1(f01,f0,env1,env,ape1,ape,ran):
    newf0 = modify(f01,f0,ran)
    newenv = np.full([env.shape[0],env.shape[1]],env)
    for k in range(env.shape[1]):
        newenv[:,k] = modify(env1[:,k],env[:,k],ran)
    
    newape = np.full([ape.shape[0],ape.shape[1]],env)
    for k in range(ape.shape[1]):
        newape[:,k] = modify(ape1[:,k],ape[:,k],ran)
    return newf0,newenv,newape


# In[9]:


def connect(f01,f02,env1,env2,ape1,ape2):
    f0 = np.hstack((f01,f02))
    env = np.vstack((env1,env2))
    ape = np.vstack((ape1,ape2))
    return f0,env,ape


# In[3]:


'''
def makepitch(f0,pitch):#MIDIノートを基準とする音程変換
    logf0 = numpy.log10(f0+0.1)
    d = (12 * ((logf0[int(len(f0)/2)] - math.log10(440))/math.log10(2)))
    d = round(d)
    newlogf0 = logf0 + (pitch - 69 - d) / 12 * math.log10(2)
    newf0 = 10 ** newlogf0
    return newf0
'''


# In[ ]:


def makePitch(f0, pitch):#constant F0
    targetF0 = 440 * pow(2, (pitch - 69) / 12)
    newF0 = np.vectorize(lambda x: 0 if x == 0 else targetF0)(f0)
    return newF0.astype(np.double)


# In[1]:


def linear(data1,data2,point):#関数leng内で使用
    data = (data2 - data1) * point + data1
    if(point < 0 or point > 1):
        sys.stderr.write("please input 0 < point < 1\n")
    return data


# In[6]:


def leng(data,f0,env,ape,time,fs):#音の長さをtime秒にする
    oritime = len(data)/fs
    x = oritime / time
    
    e1,e2 = env.shape
    a1,a2 = ape.shape
    
    newf0 = np.zeros(math.floor((len(f0) - 1.0) / x))
    newenv = np.zeros((math.floor((e1 - 1.0) / x), e2))
    newape = np.zeros((math.floor((a1 - 1.0) / x), a2))
    
    ind = 0
    
    for k in range(len(newf0)):
        f,d = math.modf(k * x)
        
        for l in range(e2):
            newenv[k,l] = linear(env[math.floor(ind),l], env[math.ceil(ind),l], f)
            newape[k,l] = linear(ape[math.floor(ind),l], ape[math.ceil(ind),l], f)
            
        newf0[k] = linear(f0[math.floor(ind)], f0[math.ceil(ind)], f)
        
        ind = ind + x
        
    return newf0,newenv,newape


# In[12]:


def hokan2(f01,f02,env1,env2,ape1,ape2,frame):
    ref01 = np.zeros(len(f01)-frame)
    reenv1 = np.zeros((env1.shape[0]-frame,env1.shape[1]))
    reape1 = np.zeros((ape1.shape[0]-frame,ape1.shape[1]))
    
    for k in range(len(ref01)):
        ref01[k] = f01[k]
        reenv1[k,:] = env1[k,:]
        reape1[k,:] = ape1[k,:]
        
    f0_hokan = np.zeros(frame)
    env_hokan = np.zeros((frame,env1.shape[1]))
    ape_hokan = np.zeros((frame,ape1.shape[1]))
    
    for k in range(frame):
        f0_hokan[k] = fi(ref01[len(ref01)-1],f02[int(len(f02)/2)],frame,k)
        for l in range(env1.shape[1]):
            env_hokan[k,l] = fi(reenv1[reenv1.shape[0]-1,l],env2[int(env2.shape[0]/2),l],frame,k)
            ape_hokan[k,l] = fi(reape1[reape1.shape[0]-1,l],ape2[int(ape2.shape[0]/2),l],frame,k)
        
    
    newf0 = np.hstack((ref01,f0_hokan))
    newf0 = np.hstack((newf0,f02))
    newenv = np.vstack((reenv1,env_hokan))
    newenv = np.vstack((newenv,env2))
    newape = np.vstack((reape1,ape_hokan))
    newape = np.vstack((newape,ape2))
        
    return newf0,newenv,newape


# In[1]:


def fall(f01,f02,fallframe,hokanframe,hnum):#現在hokan2にのみ対応,hnum:補間の種類（なし（０）、１，２）
    prep = np.zeros(fallframe)
    under = np.zeros(fallframe)
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


# In[7]:


def shakuri(f01,f02,fallframe,hokanframe,hnum):
    prep = np.zeros(fallframe)
    over = np.zeros(fallframe)
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


# In[ ]:


def extract(data, fs):  # f0, メルケプ, bapを抽出する関数
    _f0, t = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, _f0, t, fs)  # 基本周波数の修正
    sp = pw.cheaptrick(data, f0, t, fs)  # スペクトル包絡の抽出
    order = 60
    alpha = 0.544
    mcep = ps.conversion.sp2mc(sp, order, alpha)  # メルケプストラム
    ap = pw.d4c(data, f0, t, fs)  # 非周期性指標の抽出
    bap = np.zeros((len(t), 5))
    bap[:, 0] = np.average(ap[:, : 128])
    bap[:, 1] = np.average(ap[:, 128: 256])
    bap[:, 2] = np.average(ap[:, 256: 512])
    bap[:, 3] = np.average(ap[:, 512: 768])
    bap[:, 4] = np.average(ap[:, 768: 1025])  # bap
    return f0, mcep, bap


# In[ ]:


def audiomake(f0, mcep, bap, fs):
    alpha = 0.544
    fftlen = 2048
    sp = ps.conversion.mc2sp(mcep, alpha, fftlen)
    ap = np.zeros((len(f0), 1025))
    for k in range(128):
        ap[:, k] = bap[:, 0]
    for k in range(128):
        ap[:, 128+k] = bap[:, 1]
    for k in range(256):
        ap[:, 256+k] = bap[:, 2]
    for k in range(256):
        ap[:, 512+k] = bap[:, 3]
    for k in range(257):
        ap[:, 768+k] = bap[:, 4]  # bap
    synthesized = pw.synthesize(f0, sp, ap, fs)
    return synthesized


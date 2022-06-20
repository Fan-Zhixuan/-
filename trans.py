import numpy as np
import cv2
import pandas as pd
import os
num=np.arange(0,32,2)
print(num)
for i in num:
    pathH='iturHFLQ信道下信噪比'+str(i)+'H.xlsx'
    pathHall='iturHFLQ信道下信噪比'+str(i)+'Hall.xlsx'

    H= pd.read_excel(pathH,header=None)
    H_np = np.array(H)
    Hall= pd.read_excel(pathHall,header=None)
    Hall_np = np.array(Hall)
    os.makedirs(str(i))
    print(H_np.shape[0])
    assert H_np.shape==Hall_np.shape
    assert H_np.shape[0]%8==0
    long = H_np.shape[0]/8
    for j in range(int(long)):
        np.save(str(i)+'/H_'+str(j)+'.npy',H_np[8*j:8*j+8,:])
        np.save(str(i)+'/Hall_'+str(j)+'.npy',Hall_np[8*j:8*j+8,:])
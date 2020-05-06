# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:40:38 2020

"""

import pickle
import numpy as np
import pandas as pd
f = open('sample.binaryfile','rb')
data = pickle.load(f)
ID_number = 2
df_concat = pd.DataFrame(index=[])
for ID_number in range(0,15):
    Id = "ID" + str(ID_number)
    Id_x = Id + "_x"
    Id_y = Id + "_y"
    IDpos = data[ID_number]
    #%matplotlib auto

    timeseries=np.array(IDpos)
    if len(timeseries) == 0:
        continue
    df = pd.DataFrame(timeseries)
    df_=df.set_index(0)
    df_=df_.reindex(range(0,900))
    df_[Id_x] = (df_[1] + df_[3])/2
    df_[Id_y] = df_[4]
    df_concat = pd.concat([df_concat, df_])
#%%
import matplotlib.pyplot as plt
for frame in range(0,900):
    for id in range(1,10):
        str_id = "ID" + str(id)
        str_x = str_id + "_x"
        str_y = str_id + "_y"
        x = df_concat[str_x].iloc[frame]
        y = df_concat[str_y].iloc[frame]
        [[x,y]]
        plt.scatter(x,y,c='gray')
#%%
plt.scatter(df_concat["ID1_x"],df_concat["ID1_y"])
plt.scatter(df_concat["ID2_x"],df_concat["ID2_y"])
plt.scatter(df_concat["ID3_x"],df_concat["ID3_y"])
plt.scatter(df_concat["ID4_x"],df_concat["ID4_y"])
plt.scatter(df_concat["ID11_x"],df_concat["ID11_y"])
#%%
cols = [Id]
df = pd.DataFrame(index=[], columns=cols)
for t in range(0,900):
    tmp = timeseries[timeseries[:,0] == t]
    if len(tmp) == 0:
        continue
    x_cen = (tmp[0,1] + tmp[0,3])/2
    y_cen = tmp[0,4]
    df.iloc[t]=(x_cen,y_cen)
#%%

#%%
#左上が原点(0, 0) 右方向にx軸，下方向にy軸が伸びる．
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import cv2
import math
#IDpos = data[ID_number]
#tmp_pos = IDpos[]
#tmp = (frame, xl,yl,xr,yr)
ID_number = 2
IDpos = data[ID_number]
#%matplotlib auto
timeseries=np.array(IDpos)
fig = plt.figure()
ims = []
src_pts = np.array([[425, 299], [322, 296], [1766, 752], [1681, 815]], dtype=np.float32)
dst_pts = np.array([[11.5*100, 0], [8.5*100, 0], [11.5*100, 40*100], [8.5*100, 40*100]], dtype=np.float32)
mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
x_fixed = 0
y_fixed = 0
total_d = 0
pos= np.empty(3).dtype
for t in range(len(timeseries)):
    x_fixed_tmp = x_fixed
    y_fixed_tmp = y_fixed
    x_cen = (timeseries[t,1] + timeseries[t,3])/2
    #print(x_cen)
    y_cen = timeseries[t,4]
    #print(y_cen)
    tmp = np.array([[x_cen,y_cen]], dtype='float32')
    tmp = np.array([tmp])
    pts_trains = cv2.perspectiveTransform(tmp, mat)
    x_fixed = pts_trains[0,0,1]
    y_fixed = pts_trains[0,0,0]
    #pos = np.append(pos,[[x_fixed,y_fixed]],axis=0)
    if t>0:
        total_d += math.sqrt((x_fixed - x_fixed_tmp)**2 + (y_fixed - y_fixed_tmp)**2)

    #print(pts_trains)
    im = plt.scatter(x_fixed, y_fixed, c='gray', marker='.')
    ims.append([im])
plt.xlim(-50,4050)
plt.ylim(-50,2050)
plt.axes().set_aspect('equal')
ani = animation.ArtistAnimation(fig, ims, interval=1, repeat_delay=1000)
#fig.show()
#ani.save("test.gif")
#HTML(ani.to_html5_video())
#plt.plot(timeseries[:,3],timeseries[:,4])
#plt.xlim(0,1919)
#plt.ylim(0,1079)
#%%
text = "tracking_ID2_Frame274.jpg"
import re
num = re.findall( r'\d+', text)
print("ID:",num[0],"Frame:",num[1])#

#%%
ID_number = 2
Frame_cnt = 3

text = "output/tracking_ID" + str(ID_number) + "_Frame" + str(Frame_cnt) + ".jpg"
img.read(text)
#%%
import cv2
h=10
w=30
#dst_pts = np.array([[3, 0], [0, 0], [3, 40], [0, 40]], dtype=np.float32)
#dst_pts = np.array([[11.5, 0], [8.5, 0], [11.5, 40], [8.5, 40]], dtype=np.float32)
src_pts = np.array([[425, 299], [322, 296], [1766, 752], [1681, 815]], dtype=np.float32)
dst_pts = np.array([[11.5*100, 0], [8.5*100, 0], [11.5*100, 40*100], [8.5*100, 40*100]], dtype=np.float32)

mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
print(mat)
plt.scatter(src_pts[:,0],src_pts[:,1])
plt.scatter(dst_pts[:,0],dst_pts[:,1])
plt.xlim(0,2000)
plt.ylim(0,2000)
#%%
tmp = np.array([[810,311]], dtype='float32')
tmp = np.array([tmp])
pts_trains = cv2.perspectiveTransform(tmp, mat)
print(pts_trains)
#perspective_img = cv2.warpPerspective(img, mat, (w, h))
#cv2.imwrite('data/dst/opencv_perspective_dst.jpg', perspective_img)
#%%
dst_pts2 = np.array([[425, 299], [322, 296], [1766, 752], [1681, 815],[787,447]], dtype=np.float32)
plt.scatter(dst_pts2[:,0],dst_pts2[:,1])
#%%
im = cv2.imread('AAA.jpg')
perspective_img = cv2.warpPerspective(im, mat, (10000, 10000))
cv2.imwrite('opencv_perspective_dst.jpg', perspective_img)
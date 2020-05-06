# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:40:38 2020

"""

import pickle
f = open('sample.binaryfile','rb')
data = pickle.load(f)

#%%
#左上が原点(0, 0) 右方向にx軸，下方向にy軸が伸びる．
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
#IDpos = data[ID_number]
#tmp_pos = IDpos[]
#tmp = (frame, xl,yl,xr,yr)
ID_number = 2
IDpos = data[ID_number]
%matplotlib auto
timeseries=np.array(IDpos)
fig = plt.figure()
ims = []
for t in range(len(timeseries)):
    im = plt.scatter(timeseries[t,3], timeseries[t,4], c='gray', marker='.')
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, repeat_delay=1000)
fig.show()
ani.save("test.gif")
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
import sys
import glob
import cv2
import numpy as np
import os

from shutil import copytree
from shutil import rmtree

categories=['iranKhodro_dena', 'kia_cerato', 'mazda_3', 'peugeot_206', 'saipa_saina']
default_path=".\..\data"
ttv = ['train', 'test', 'valid']

for i in ttv:
  os.mkdir(".\..\data\{}".format(i))

whole_images=[]
for i in categories:
  images_path=os.path.join(default_path, i)
  images = glob.glob(images_path + "\*.jpg")
  images.sort()
  whole_images.append(images)

for i in ttv:
  for j in categories:
    os.mkdir(".\..\data\{}\{}".format(i, j))

for i in range(len(ttv)):
  for j in range(len(categories)):
    if i ==2:
      count=len(whole_images[j][int(len(whole_images[j])*0.1)+5:])+5
      for k in range(int(len(whole_images[j])*0.1)):
        img=cv2.imread(whole_images[j][count+k])
        img_name=os.path.split(whole_images[j][count+k])[1]
        cv2.imwrite(".\..\data\{}\{}\{}".format(ttv[i], categories[j], img_name), img)
        os.remove(whole_images[j][count+k])
    elif i ==1:
      count=len(whole_images[j][int(len(whole_images[j])*0.1)+5:])
      for k in range(5):
        img=cv2.imread(whole_images[j][count+k])
        img_name=os.path.split(whole_images[j][count+k])[1]
        cv2.imwrite(".\..\data\{}\{}\{}".format(ttv[i], categories[j], img_name), img)
        os.remove(whole_images[j][count+k])
    else:
      for k in range(len(whole_images[j][int(len(whole_images[j])*0.1)+5:])):
        img=cv2.imread(whole_images[j][k])
        img_name=os.path.split(whole_images[j][k])[1]
        cv2.imwrite(".\..\data\{}\{}\{}".format(ttv[i], categories[j], img_name), img)    
        os.remove(whole_images[j][k])
		
for i in categories:
    rmtree(".\..\data\{}".format(i))
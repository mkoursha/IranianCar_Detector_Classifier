import sys
import glob
import cv2
import numpy as np
import os

from shutil import copytree

copytree(".\\..\\data", ".\\..\\resized_data")

categories=['iranKhodro_dena', 'kia_cerato', 'mazda_3', 'peugeot_206', 'saipa_saina']
default_path=".\\..\\resized_data"
ttv = ['train', 'test', 'valid']


for j in ttv:
    for i in categories:
      images_path="{}\\{}\\{}".format(default_path, j, i)
      images = glob.glob(images_path + "\\*.jpg")
      images.sort()
      for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (416, 416))
        os.remove(img_path)
        cv2.imwrite(img_path, img)
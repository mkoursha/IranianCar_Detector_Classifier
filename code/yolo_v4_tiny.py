# -*- coding: utf-8 -*-
"""YOLO-v4-tiny.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cfzFRslKbx0kP8SatTCncpvyK4lvi44H
"""

"""this file is the .py file of .ipynb file
for running please run the .ipynb file"""

!apt-get install gti-lfs

!git lfs clone https://github.com/mkoursha/IranianCar_Detector_Classifier.git

import tensorflow as tf

# Commented out IPython magic to ensure Python compatibility.
!/usr/local/cuda/bin/nvcc --version

!nvidia-smi

# %cd /content/
# %rm -rf darknet

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/roboflow-ai/darknet.git

# %cd darknet/
!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
!sed -i 's/GPU=0/GPU=1/g' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
!sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= -gencode arch=compute_${compute_capability},code=sm_${compute_capability}/g" Makefile
!make

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/darknet
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

# Commented out IPython magic to ensure Python compatibility.
# %cp -r /content/IranianCar_Detector_Classifier/resized_data/. /content/darknet/
# %cp train/classes.txt data/obj.names
# %mkdir data/obj
# %cp train/*.jpg data/obj/
# %cp valid/*.jpg data/obj/

# %cp train/*.txt data/obj/
# %cp valid/*.txt data/obj/

with open('data/obj.data', 'w') as out:
  out.write('classes = 5\n')
  out.write('train = data/train.txt\n')
  out.write('valid = data/valid.txt\n')
  out.write('names = data/obj.names\n')
  out.write('backup = backup/')

# Commented out IPython magic to ensure Python compatibility.
import os

with open('data/train.txt', 'w') as out:
  for img in [f for f in os.listdir('train') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

with open('data/valid.txt', 'w') as out:
  for img in [f for f in os.listdir('valid') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')


# %ls /content/IranianCar_Detector_Classifier/train/*.txt

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

num_classes = file_len('train/classes.txt')
max_batches = num_classes*2000
steps1 = .8 * max_batches
steps2 = .9 * max_batches
steps_str = str(steps1)+','+str(steps2)
num_filters = (num_classes + 5) * 3

print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

if os.path.exists('./cfg/custom-yolov4-tiny-detector.cfg'): os.remove('./cfg/custom-yolov4-tiny-detector.cfg')

from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

# Commented out IPython magic to ensure Python compatibility.
# %%writetemplate ./cfg/custom-yolov4-tiny-detector.cfg
# [net]
# # Testing
# #batch=1
# #subdivisions=1
# # Training
# batch=64
# subdivisions=16
# width=416
# height=416
# channels=3
# momentum=0.9
# decay=0.0005
# angle=0
# saturation = 1.5
# exposure = 1.5
# hue=.1
# 
# learning_rate=0.00261
# burn_in=1000
# max_batches = {max_batches}
# policy=steps
# steps={steps_str}
# scales=.1,.1
#  
# [convolutional]
# batch_normalize=1
# filters=32
# size=3
# stride=2
# pad=1
# activation=leaky
# 
# [convolutional]
# batch_normalize=1
# filters=64
# size=3
# stride=2
# pad=1
# activation=leaky
#  
# [convolutional]
# batch_normalize=1
# filters=64
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers=-1
# groups=2
# group_id=1
#  
# [convolutional]
# batch_normalize=1
# filters=32
# size=3
# stride=1
# pad=1
# activation=leaky
#  
# [convolutional]
# batch_normalize=1
# filters=32
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers = -1,-2
# 
# [convolutional]
# batch_normalize=1
# filters=64
# size=1
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers = -6,-1
# 
# [maxpool]
# size=2
# stride=2
# 
# [convolutional]
# batch_normalize=1
# filters=128
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers=-1
# groups=2
# group_id=1
# 
# [convolutional]
# batch_normalize=1
# filters=64
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [convolutional]
# batch_normalize=1
# filters=64
# size=3
# stride=1
# pad=1
# activation=leaky
#  
# [route]
# layers = -1,-2
# 
# [convolutional]
# batch_normalize=1
# filters=128
# size=1
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers = -6,-1
# 
# [maxpool]
# size=2
# stride=2
# 
# [convolutional]
# batch_normalize=1
# filters=256
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers=-1
# groups=2
# group_id=1
# 
# [convolutional]
# batch_normalize=1
# filters=128
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [convolutional]
# batch_normalize=1
# filters=128
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers = -1,-2
# 
# [convolutional]
# batch_normalize=1
# filters=256
# size=1
# stride=1
# pad=1
# activation=leaky
# 
# [route]
# layers = -6,-1
# 
# [maxpool]
# size=2
# stride=2
# 
# [convolutional]
# batch_normalize=1
# filters=512
# size=3
# stride=1
# pad=1
# activation=leaky
#  
# # ##################################
# 
# [convolutional]
# batch_normalize=1
# filters=256
# size=1
# stride=1
# pad=1
# activation=leaky
#  
# [convolutional]
# batch_normalize=1
# filters=512
# size=3
# stride=1
# pad=1
# activation=leaky
#  
# [convolutional]
# size=1
# stride=1
# pad=1
# filters={num_filters}
# activation=linear
# 
# 
# 
# [yolo]
# mask = 3,4,5
# anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
# classes={num_classes}
# num=6
# jitter=.3
# scale_x_y = 1.05
# cls_normalizer=1.0
# iou_normalizer=0.07
# iou_loss=ciou
# ignore_thresh = .7
# truth_thresh = 1
# random=0
# nms_kind=greedynms
# beta_nms=0.6
# 
# [route]
# layers = -4
#  
# [convolutional]
# batch_normalize=1
# filters=128
# size=1
# stride=1
# pad=1
# activation=leaky
# 
# [upsample]
# stride=2
# 
# [route]
# layers = -1, 23
# 
# [convolutional]
# batch_normalize=1
# filters=256
# size=3
# stride=1
# pad=1
# activation=leaky
# 
# [convolutional]
# size=1
# stride=1
# pad=1
# filters={num_filters}
# activation=linear
# 
# [yolo]
# mask = 1,2,3
# anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
# classes={num_classes}
# num=6
# jitter=.3
# scale_x_y = 1.05
# cls_normalizer=1.0
# iou_normalizer=0.07
# iou_loss=ciou
# ignore_thresh = .7
# truth_thresh = 1
# random=0
# nms_kind=greedynms
# beta_nms=0.6

# Commented out IPython magic to ensure Python compatibility.
# %cat cfg/custom-yolov4-tiny-detector.cfg

!./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map

# Commented out IPython magic to ensure Python compatibility.
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
#   %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

!ls backup

# Commented out IPython magic to ensure Python compatibility.
!cp /content/darknet/backup/custom-yolov4-tiny-detector_final.weights "/content/drive/My Drive"
!cp /content/darknet/backup/custom-yolov4-tiny-detector_best.weights "/content/drive/My Drive"

# %cp data/obj.names data/coco.names

# Commented out IPython magic to ensure Python compatibility.
test_images = [f for f in os.listdir('test') if f.endswith('.jpg')]
import random
img_path = "test/" + random.choice(test_images);

# %cd /content/darknet/
img_path = "/content/darknet/valid/3333361.jpg"
!./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights {img_path} -dont-show
imShow('predictions.jpg')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git
# %cd /content/tensorflow-yolov4-tflite

!cp /content/darknet/data/obj.names /content/tensorflow-yolov4-tflite/data/classes/
!ls /content/tensorflow-yolov4-tflite/data/classes/

!sed -i "s/coco.names/obj.names/g" /content/tensorflow-yolov4-tflite/core/config.py

# %cd /content/tensorflow-yolov4-tflite

# Commented out IPython magic to ensure Python compatibility.
!python save_model.py \
  --weights /content/darknet/backup/custom-yolov4-tiny-detector_final.weights \
  --output ./checkpoints/yolov4-tiny-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \

!python save_model.py \
  --weights /content/darknet/backup/custom-yolov4-tiny-detector_best.weights \
  --output ./checkpoints/yolov4-tiny-pretflite-416 \
  --input_size 416 \
  --model yolov4 \
  --tiny \
  --framework tflite


# %cd /content/tensorflow-yolov4-tflite
!python convert_tflite.py --weights ./checkpoints/yolov4-tiny-pretflite-416 --output ./checkpoints/yolov4-tiny-416.tflite

!ls /content/darknet/test

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/tensorflow-yolov4-tflite
!python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 \
  --image /content/darknet/valid/3333361.jpg \
  # --framework tflite

# %cd /content/tensorflow-yolov4-tflite/
!ls
from IPython.display import Image
Image('/content/tensorflow-yolov4-tflite/result.png')


!cp -r /content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416/ "/content/drive/My Drive"

!cp /content/tensorflow-yolov4-tflite/checkpoints/yolov4-tiny-416.tflite "/content/drive/My Drive"

!cp -r /content/tensorflow-yolov4-tflite/* "/content/drive/My Drive/tensorflow-yolov4-tflite"

!cp -r /content/darknet/* "/content/drive/My Drive/darknet"
import os

import glob
import cv2

from shutil import rmtree, copyfile

tv = ['train', 'valid']
categories = ['iranKhodro_dena', 'kia_cerato', 'mazda_3', 'peugeot_206', 'saipa_saina']

base_path = "..\\resized_data"


jpg_files = []
txt_files = []

copyfile("..\\resized_data\\train\\iranKhodro_dena\\classes.txt", "..\\resized_data\\train\\classes.txt")

for parent_dir in ['test', 'train', 'valid']:
  for category in categories:
    files_path = os.path.join(base_path, parent_dir)
    files_path = os.path.join(files_path, category)
    os.remove(files_path+'\\classes.txt')

for parent_dir in tv:
  for category in categories:
    files_path = os.path.join(base_path, parent_dir)
    files_path = os.path.join(files_path, category)
    files = glob.glob(files_path+'\\*.jpg')
    files.sort()
    jpg_files.append(files)
    files = glob.glob(files_path+'\\*.txt')
    files.sort()
    txt_files.append(files)


for jpg_list in jpg_files:
  for jpg_file in jpg_list:
    category = jpg_file.split('\\')[-2]
    file_name = jpg_file.split('\\')[-1]
    img = cv2.imread(jpg_file)
    file_name = (str(categories.index(category))*10) + file_name
    target_path = "\\".join(jpg_file.split('\\')[:-2]) + "\\{}".format(file_name)
    cv2.imwrite(target_path, img)
    os.remove(jpg_file)


for txt_list in txt_files:
  for txt_file in txt_list:
    category = txt_file.split('\\')[-2]
    file_name = txt_file.split('\\')[-1]
    f = open(txt_file, 'r')
    lines = f.readlines()
    f.close()
    file_name = (str(categories.index(category))*10) + file_name
    target_path = "\\".join(txt_file.split('\\')[:-2]) + "\\{}".format(file_name)
    f = open(target_path, 'w')
    f.writelines(lines)
    f.close()
    os.remove(txt_file)

for parent_dir in tv:
  for category in categories:
    files_path = os.path.join(base_path, parent_dir)
    files_path = os.path.join(files_path, category)
    rmtree(files_path)

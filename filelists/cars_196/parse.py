import scipy.io as scio
import pandas as pd
import os
import shutil
from os.path import isfile, isdir, join
 
cwd = os.getcwd() 
data_path = join(cwd,'car_ims/')
print('data_path:', data_path)

savedir = join(cwd,'cars_196_images/images/')
try:
    os.stat(savedir)
except:
    os.makedirs(savedir)








for clas in range(196):

    class_dir = join(savedir, str(clas + 1))

    try:
        os.stat(class_dir)
    except:
        os.makedirs(class_dir)





path = './cars_annos.mat'
data = scio.loadmat(path)
annos = data['annotations'] 
image_num = len(annos[0])

for i in range(image_num):
    image_name = str(annos[0][i][0])[2:-2].split("/")[1]
    image_class = str(annos[0][i][-2])[2:-2]
    # print(data_path + image_name, savedir + image_class)
    shutil.copy(data_path + image_name, savedir + image_class)





print('image_num:', image_num)
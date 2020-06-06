
from os import listdir
import shutil
from os.path import isfile, isdir, join
import os


cwd = os.getcwd() 
data_path = join(cwd,'fgvc-aircraft-2013b/data/')
print('data_path:', data_path)

savedir = join(cwd,'aircraft/images/')
try:
    os.stat(savedir)
except:
    os.makedirs(savedir)


classes_file = open(data_path + 'variants.txt', 'r')
classes = classes_file.readlines()
# print(classes)

for clas in classes:

    clas = clas.strip().replace(" ", "_");

    class_dir = join(savedir, clas)

    try:
        os.stat(class_dir)
    except:
        os.makedirs(class_dir)


# train_dataset
trainval_file = open(data_path + 'images_variant_trainval.txt', 'r')
trainval = trainval_file.readlines()
# print(trainval)
train_images_count = 0
for item in trainval:

    image_name = item.strip().split(" ")[0] + '.jpg'
    class_name = item.strip().split(" ", 1)[-1].replace(" ", '_')
    train_images_count += 1
    print(data_path + 'images/' + image_name, savedir + class_name)
    shutil.copy(data_path + 'images/' + image_name, savedir + class_name)
print('train_count:', train_images_count)



# test_dataset
test_file = open(data_path + 'images_variant_test.txt', 'r')
test = test_file.readlines()
# print(trainval)
test_images_count = 0
for item in test:

    image_name = item.strip().split(" ")[0] + '.jpg'
    class_name = item.strip().split(" ", 1)[-1].replace(" ", '_')
    test_images_count += 1
    print(data_path + 'images/' + image_name, savedir + class_name)
    shutil.copy(data_path + 'images/' + image_name, savedir + class_name)
print('test_count:', test_images_count)
total_count = train_images_count + test_images_count
print('total_count:', total_count)




    # print(clas)





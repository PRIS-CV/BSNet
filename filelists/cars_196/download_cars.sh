#!/usr/bin/env bash
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
tar -zxvf car_ims.tgz
python parse.py
python write_cars_filelist.py

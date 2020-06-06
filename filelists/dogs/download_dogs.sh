#!/usr/bin/env bash
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
tar -xvf images.tar
tar -xvf annotation.tar
python write_dogs_filelist.py

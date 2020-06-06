#!/usr/bin/env bash
wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -zxvf fgvc-aircraft-2013b.tar.gz
python classifier.py
python write_aircraft_filelist.py

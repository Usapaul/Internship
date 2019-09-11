#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as units
from dustmaps.bayestar import BayestarWebQuery


def readdata(numrows=100000):
	array = []
	for i in range(numrows):
	        array.append(f.readline().split())
	idlbd = []
	for i in range(numrows):
	    idlbd.append([array[i][0],array[i][1],array[i][2],array[i][3]])
	global source_id
	for i in range(numrows):
		source_id.append(int(idlbd[i][0]))
		idlbd[i][0] = int(idlbd[i][0])
		idlbd[i][1] = float(idlbd[i][1])
		idlbd[i][2] = float(idlbd[i][2])
		idlbd[i][3] = float(idlbd[i][3])
	return idlbd


data_filename = 'full_HRD_100k_for_example_idlbd'
num_rows_in_data_file = 100000


f = open(data_filename+'.txt','r')
f.readline()

num_param_red = 1
resultarray = np.zeros((100000,num_param_red))
bayestar = BayestarWebQuery(version='bayestar2017')

fout = open(data_filename+'_SFD_b17.txt','w')

num_of_sets = (num_rows_in_data_file - 1) // 100000 + 1

for numset in range(num_of_sets):
	numrows = 100000
	if numset == num_of_sets - 1:
		numrows = (num_rows_in_data_file - 1) % 100000 + 1
		resultarray = np.zeros((numrows,num_param_red))
	source_id = []
	workarray = np.array(readdata(numrows))
	l = workarray[:,1] * units.deg
	b = workarray[:,2] * units.deg
	d = workarray[:,3] * units.pc
	coords = SkyCoord(l, b, distance=d, frame='galactic')
	resultarray = bayestar(coords,mode='mean')
	for i in range(numrows):
		line2file = str(source_id[i]) + ' ' + str(resultarray[i])
		fout.write(line2file+'\n')
	print(str(numset+1)+'/'+str(num_of_sets))


f.close()
fout.close()

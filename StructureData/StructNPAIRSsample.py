#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:18:33 2017

@author: xiaomuliu
"""

import numpy as np
import pandas as pd
import re
import cPickle as pickle
from StructClusterData import point_data_subset, flatten_binned_2darray
from sklearn.model_selection import KFold
import sys
sys.path.append('..')
from Misc.ComdArgParse import ParseArg

args = ParseArg()
infiles = args['input']
outfile = args['output']
params = args['param']

#convert string input to tuple
param_match1 = re.search('(?<=daterange=)(\d{4}-\d{2}-\d{2}\s*){2}', params)
param_match2 = re.search('(?<=featurecrimetypes=)([A-Za-z_\s]+)(?=Nsamples)', params)
Nsamples = int(re.search('(?<=Nsamples=)(\d+)',params).group(1))
r_seed = int(re.search('(?<=initRseed=)(\d+)',params).group(1))

infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)
  
grid_pkl = infile_match.group(1)
crimedata_path = infile_match.group(2)

date_range = param_match1.group(0).rstrip().split(' ')
FeatureCrimeTypes = re.findall('[A-Za-z_]+',param_match2.group(0))
         
CrimeData_list = []
for crimetype in FeatureCrimeTypes:        
    fileName_load = crimedata_path + crimetype + "_08_14.pkl"
    with open(fileName_load,'rb') as input_file:
        CrimeData_list.append(point_data_subset(pickle.load(input_file),date_range=date_range,coord_only=False))

# load grid info  
with open(grid_pkl,'rb') as grid_file:
    grid_list = pickle.load(grid_file)
_, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
grid_2d = (grd_x, grd_y)
Ngrids = np.nansum(mask_grdInCity) 

# initialize a list to save splited samples where each element is a dict with keys in ('train' 'test')
# and values are corresponding splited feature arrays
split_samples = [None]*Nsamples

# split data according to dates
date_seq = pd.date_range(date_range[0], date_range[1], freq='D')
date_idx = range(len(date_seq))
for i in range(Nsamples):
    two_folds = KFold(n_splits=2, shuffle=True, random_state=r_seed+i) # change rand_state in every iteration
    for s1, s2 in two_folds.split(date_idx):
        date_train, date_test = date_seq[s1], date_seq[s2]
        X_train = np.zeros((Ngrids,len(CrimeData_list)))
        X_test = np.zeros((Ngrids,len(CrimeData_list)))
        
        for j, point_data in enumerate(CrimeData_list):
            f_train = point_data_subset(point_data,date_list=date_train)
            f_test = point_data_subset(point_data,date_list=date_test)
            
            # bin crime data of each type
            X_train[:,j] = flatten_binned_2darray(f_train.values, grid_2d, cellsize)[mask_grdInCity]
            X_test[:,j] = flatten_binned_2darray(f_test.values, grid_2d, cellsize)[mask_grdInCity]

    split_samples[i] = {'train':X_train,'test':X_test}  

# *********************** Save objects ******************* #                              
with open(outfile,'wb') as output:
    pickle.dump(split_samples, output, pickle.HIGHEST_PROTOCOL)
   
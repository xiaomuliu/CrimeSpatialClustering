#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=======================================
Binned Crime Count Count Feature Array
=======================================
Created on Wed Oct  5 11:33:40 2016

@author: xiaomuliu
"""
import numpy as np
from datetime import datetime
import cPickle as pickle
import sys
sys.path.append('..')
from ImageProcessing.KernelSmoothing import bin_point_data_2d


def point_data_subset(data, date_range=None, date_list=None, coord_only=True):
    """
    Return a subset of crime point data given a specified lists of dates or a specified date or year/month range. 
    date_range is a list-like object containing two string elements of which the first one 
    specifies the beginning date while the second one specifies the end date. 
    The date string should be in form of 'YYYY-MM-DD'
    """
    
    if date_range is not None:
        date_range = [datetime.strptime(d,'%Y-%m-%d').date() for d in date_range]          
        data_sub = data.ix[(data['DATEOCC']>=date_range[0]) & (data['DATEOCC']<=date_range[1]),:]
    else:
        data_sub = data
        
    if date_list is not None:
        date_list = [datetime.strptime(d,'%Y-%m-%d').date() if isinstance(d, str) else d.date() for d in date_list]            
        data_sub = data.ix[np.in1d(data['DATEOCC'].values,date_list),:]
    else:
        data_sub = data
        
    if coord_only:
        data_sub = data_sub[['X_COORD','Y_COORD']]

    return data_sub

 
def flatten_binned_2darray(points, grid, cellsize=None):
    """
    Return 1D vector of binned point counts
    """
    if cellsize is None:
        # assuming a regular grid with equal cell size, the cellsize is of [size_x, size_y]
        cellsize = (np.abs(np.diff(grid[0][:2])), np.abs(np.diff(grid[1][:2])))
    
    binned_data = bin_point_data_2d(points, grid, cellsize, stat='count', geoIm=False)   
    flattened_data = binned_data.ravel(order='F')
              
    return flattened_data     
    
    
if __name__=='__main__':  
    import pandas as pd
    import re
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infiles = args['input']
    outfile = args['output']
    params = args['param']

    #convert string input to tuple
    param_match1 = re.search('(?<=daterange=)(\d{4}-\d{2}-\d{2}\s*){2}', params)
    param_match2 = re.search('(?<=featurecrimetypes=)([A-Za-z_\s]+)', params)
    
    infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)
      
    grid_pkl = infile_match.group(1)
    crimedata_path = infile_match.group(2)
    
    date_range = param_match1.group(0).rstrip().split(' ')
    FeatureCrimeTypes = re.findall('[A-Za-z_]+',param_match2.group(0))

             
    CrimeData_list = []
    for crimetype in FeatureCrimeTypes:        
        fileName_load = crimedata_path + crimetype + "_08_14.pkl"
        with open(fileName_load,'rb') as input_file:
            CrimeData_list.append(point_data_subset(pickle.load(input_file),date_range))

    # load grid info  
    with open(grid_pkl,'rb') as grid_file:
        grid_list = pickle.load(grid_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
    
    # bin crime data of each type
    hist_array = np.zeros((Ngrids,len(CrimeData_list)))
    for i, point_data in enumerate(CrimeData_list):
        hist_array[:,i] = flatten_binned_2darray(point_data.values, grid_2d, cellsize)[mask_grdInCity]

#    # verify
#    from SetupGrid import flattened_to_geoIm
#    import matplotlib.pyplot as plt
#    
#    hist_2d = flattened_to_geoIm(hist_array[:,0],len(grd_x),len(grd_y),mask=mask_grdInCity)
#
#    fig = plt.figure(figsize=(8, 6))
#    plt.imshow(hist_2d, interpolation='nearest', origin='upper', cmap='jet')
#    plt.title('Binned homicide count')
#    plt.colorbar()
    
    # *********************** Save objects ******************* #
    default_savePath = "../Clustering/FeatureData/grid_"+str(cellsize[0])+"_"+str(cellsize[1])+"/" 
    filePath_save = outfile if outfile is not None else default_savePath
       
#    feature_dict = {'FeatureArray':hist_array, 'FeatureName':FeatureCrimeTypes}                           
#    savefile_dict = filePath_save+'cluster_feature_dict.pkl'
#    with open(savefile_dict,'wb') as output:
#        pickle.dump(feature_dict, output, pickle.HIGHEST_PROTOCOL)
   
    feature_df = pd.DataFrame(hist_array,columns=FeatureCrimeTypes)
    savefile_df = filePath_save+'cluster_feature_dataframe.pkl' 
    with open(savefile_df,'wb') as output: 
        pickle.dump(feature_df, output, pickle.HIGHEST_PROTOCOL)
             
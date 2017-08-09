#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:09:49 2017

@author: xiaomuliu
"""
import numpy as np
import cPickle as pickle
import VisualizeCluster as vc
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 14
LARGE_SIZE = 16

import matplotlib
params = {'legend.fontsize': MEDIUM_SIZE,
         'axes.labelsize': MEDIUM_SIZE,
         'axes.titlesize':MEDIUM_SIZE,
         'xtick.labelsize':MEDIUM_SIZE,
         'ytick.labelsize':MEDIUM_SIZE}
matplotlib.rcParams.update(params)


def plot_cluster_stats(cluster_obj, featureNames, grid, figpath, cmap='jet'):

    Ncomp = cluster_obj['n_components']
    fig = plt.figure(figsize=(12,10)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, cluster_obj['label'], grid, flattened=True, mask=mask_grdInCity)        
    vc.colorbar_index(ncolors=Ncomp,ax=ax,cmap=cmap,shrink=0.6)  
    figname = 'HMRF_GMM_segmentation.png'    
    vc.save_figure(figpath+figname,fig)
    

    GMM_stats = [(featureNames, cluster_obj['GMM']['means_'][i,:],cluster_obj['GMM']['covariances_'][i,:,:]) for i in xrange(Ncomp)]  

    for i in xrange(Ncomp):
        fig, axes = plt.subplots(1, 3, figsize=(18,7)) 
        fig.subplots_adjust(bottom=0.4, top=0.9, wspace=0.9)
        vc.plot_GMM_stats(axes,GMM_stats[i],['mean','Cov','Corr'])
#        fig.suptitle('Cluster '+str(i),fontsize=12)
        figname = 'HMRF_Cluster'+str(i)+'_stats.png'    
        vc.save_figure(figpath+figname,fig)
            
        
if __name__=='__main__':

    path = '/Users/xiaomuliu/CrimeProject/SpatioTemporalPredictiveModeling_daily/SharedData/SpatialData/grid_500_500/'
    grid_pkl = path+'grid.pkl'    
    # load grid info   
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
        

    cluster_file = path+'cluster/clusters_MRF_GMM_Ncomp_6_beta_1_0.pkl'
     
    with open(cluster_file,'rb') as input_file:
        cluster = pickle.load(input_file)
                
    featureNames = ['Homicide', 'Sexual Assault', 'Robbery', 'Aggravated Assault', 
                    'Aggravated Battery', 'Simple Assault', 'Simple Battery', 'Burglary', 
                    'Larceny', 'Motor Vehicle Theft', 'Weapons Violation', 'Narcotics'] 
    GMM = cluster['GMM']
    Ncomp = cluster['n_components']
    beta = cluster['beta']                       
    figpath = '/Users/xiaomuliu/CrimeProject/SpatioTemporalPredictiveModeling_daily/Clustering/Figures/grid_500_500/GMM_HMRF/Ncomp_'+\
                str(Ncomp)+'/beta_'+str(beta)+'/'
    plot_cluster_stats(cluster, featureNames, grid_2d, figpath, cmap='jet')
    
    for i in xrange(Ncomp):
        print 'Component: '+str(i)
        for j,f in enumerate(featureNames):
          print f, round(GMM['means_'][i,j],2), round(np.sqrt(GMM['covariances_'][i,j,j]),2)
          
          
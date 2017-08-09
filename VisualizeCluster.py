#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
========================================
Visualize Clusters on Spatial Space
========================================
Created on Thu Oct  6 14:54:17 2016

@author: xiaomuliu
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import sys
sys.path.append('..')
from Grid.SetupGrid import flattened_to_geoIm

def plot_clusters(ax, cluster_label, grid, flattened=True, mask=None, cmap='jet'):
    grd_x, grd_y = grid
    if mask is None:
        mask = np.ones(cluster_label.shape).astype('bool')
    if flattened:
        cl_label_im = flattened_to_geoIm(cluster_label,len(grd_x),len(grd_y),mask=mask)        
    else:
        cl_label_im = cluster_label
        
    ax.matshow(cl_label_im, interpolation='nearest', origin='upper',cmap=cmap)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    #ax.set_title('Number of clusters: '+str(len(np.unique(cluster_label))))  
    
    
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    Reference: http://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/ColormapTransformations
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def colorbar_index(ncolors, ax=None, cmap=plt.rcParams['image.cmap'].encode('ascii','ignore'),**kwargs):
    """
    Create colorbar(legend) for matix/image plot such that each unique value in 
    matrix/image has a corresponding label
    The default colormap is set to rc.image.cmap
    """ 
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, ax=ax, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))    

def plot_means(ax, means, var_names=None): 
    if var_names is None:
        var_names = np.arange(len(means))
        
    ax.yaxis.grid(True, alpha=0.7)
    ax.bar(np.arange(len(means))-0.4, means, width=0.7, color='blue')
    ax.set_xlim(-.4, len(var_names) - .4)
    #ax.set_aspect(5)
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_xticklabels(var_names,rotation='vertical',horizontalalignment='center')
    ax.set_ylabel('Mean')
    ax.set_title('Means')

    
def cov2corr(Cov):
    p = Cov.shape[0]
    sd_inv = 1/np.sqrt(np.diag(Cov)).astype(float)
    return np.tile(sd_inv,(p,1)) * Cov * np.tile(sd_inv,(p,1)).T

def plot_mat(ax, Corr, var_names=None, title='', cmap='gray'):
    if var_names is None:
        var_names = np.arange(Corr.shape[0])
           
    cax = ax.imshow(Corr, interpolation='nearest', origin='upper', cmap=cmap)
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
    ax.set_xticklabels(var_names,rotation='vertical')
    ax.set_yticklabels(var_names)
    ax.set_title(title)
    plt.colorbar(cax,ax=ax,shrink=0.65) 

def plot_GMM_stats(axes, ranked_features, stats=['mean','Cov','Corr'], mat_cmap='gray'):        
    if len(axes)!=len(stats):
        raise ValueError('Number of axes must equal to the number of statistics to plot') 
    
    features, means, cov_mat = ranked_features    
    if 'mean' in stats:   
        ax = axes[np.array(stats)=='mean'].tolist()[0]
        plot_means(ax, means, features)
    if 'Cov' in stats:
        ax = axes[np.array(stats)=='Cov'].tolist()[0]
        plot_mat(ax, cov_mat, features,'Covariance matrix', mat_cmap)
    if 'Corr' in stats:
        corr_mats = cov2corr(cov_mat)
        ax = axes[np.array(stats)=='Corr'].tolist()[0]
        plot_mat(ax, corr_mats, features,'Correlation matrix', mat_cmap)    

        
def save_figure(fig_name,fig):
    if not os.path.exists(os.path.dirname(fig_name)):
        os.makedirs(os.path.dirname(fig_name))
    fig.savefig(fig_name)        
        
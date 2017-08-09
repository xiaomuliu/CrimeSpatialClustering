#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:12:58 2017

@author: xiaomuliu
"""
import matplotlib.pyplot as plt
#from matplotlib import rc
import itertools
import numpy as np
import os

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 16

import matplotlib
params = {'legend.fontsize': MEDIUM_SIZE,
          'figure.figsize': (11, 11),
         'axes.labelsize': MEDIUM_SIZE,
         'axes.titlesize':MEDIUM_SIZE,
         'xtick.labelsize':SMALL_SIZE,
         'ytick.labelsize':SMALL_SIZE}
matplotlib.rcParams.update(params)


def save_figure(fig_name,fig):
    if not os.path.exists(os.path.dirname(fig_name)):
        os.makedirs(os.path.dirname(fig_name))
    fig.savefig(fig_name)   
    
def plot_PR(P_list,R_list,models,errbar=True,errbarbound=(25, 75), markers=['.','x'], cmap='rainbow', \
            show=True, save_fig=False, fig_name='PR.png'):
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, len(P_list))]
    
    mk_styles = itertools.cycle(markers)
    
    P_medians = [np.median(p) for p in P_list]
    R_medians = [np.median(r) for r in R_list]
                 
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    if errbar:
        P_errs = [[[pm-np.percentile(p, errbarbound[0])],[np.percentile(p, errbarbound[1])-pm]] for p,pm in zip(P_list,P_medians)]
        R_errs = [[[rm-np.percentile(r, errbarbound[0])],[np.percentile(r, errbarbound[1])-rm]] for r,rm in zip(R_list,R_medians)]
        for i, (cl, mk) in enumerate(zip(colors,mk_styles)):
#            # use TeX symbol
#            lab = models[i].replace('beta',r'$\beta$')
            plt.errorbar(R_medians[i], P_medians[i], xerr=R_errs[i], yerr=P_errs[i], marker=mk, color=cl, label=models[i], capsize=4)
    else:
        for i, (cl, mk) in enumerate(zip(colors,mk_styles)):
            plt.scatter(R_medians[i], P_medians[i], marker=mk, c=cl, label=models[i])   
                        
    ax.set_ylabel('Log-likelihood')
    ax.set_xlabel('Adjusted Rand Index')        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))        

    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig)  
        
 
def plot_PR2(P_list,R_list,models,errbar=True,errbarbound=(25, 75), markers=['o','x','s'], colors=['r','g','b'], \
            show=True, save_fig=False, fig_name='PR.png'):
        
    P_medians = [np.median(p) for p in P_list]
    R_medians = [np.median(r) for r in R_list]
                
    cl_mk = list(itertools.product(colors,markers))  #outer product of two lists   
                 
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    if errbar:
        P_errs = [[[pm-np.percentile(p, errbarbound[0])],[np.percentile(p, errbarbound[1])-pm]] for p,pm in zip(P_list,P_medians)]
        R_errs = [[[rm-np.percentile(r, errbarbound[0])],[np.percentile(r, errbarbound[1])-rm]] for r,rm in zip(R_list,R_medians)]
        for i, (cl, mk) in enumerate(cl_mk):
            # use TeX symbol
            lab = models[i].replace('beta',r'$\beta$')
            plt.errorbar(R_medians[i], P_medians[i], xerr=R_errs[i], yerr=P_errs[i], marker=mk, color=cl, label=lab, capsize=4)
    else:
        for i, (cl, mk) in enumerate(cl_mk):
            plt.scatter(R_medians[i], P_medians[i], marker=mk, c=cl, label=models[i])   
                        
    ax.set_ylabel('Log-likelihood')
    ax.set_xlabel('Adjusted Rand Index')    
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))        

    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig)         

        
        

if __name__=='__main__':
    import re
    import cPickle as pickle
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg    

    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    PRfile_list = re.search('([\w\./\s;]+)',infiles).group(1).rstrip('; ').split('; ')
           
    filePath_save = outpath if outpath is not None else './NPAIRS/Evaluation/'
    
    modelparam_list = re.search('([\w\./\s=,;]+)',params).group(1).rstrip('; ').split('; ')
       
    P_list = []
    R_list = []
    for pr_file in PRfile_list:
        with open(pr_file,'rb') as input_file:
            PR = pickle.load(input_file)
        P_list.append(PR['accuracy'].ravel())    
        R_list.append(PR['reproducibility'])  
    
    # prune log-likelihood values (for those invalid values which were obtained by log(0))
    P_list_temp = P_list
    for i,p_vec in enumerate(P_list):
        P_list_temp[i] = [p for p in p_vec if np.isfinite(p)]  

    P_list = P_list_temp 

            
        
#    plot_PR(P_list,R_list,modelparam_list,errbar=False,errbarbound=(25, 75), markers=['o','x','s','^'], cmap='jet', \
#            show=False, save_fig=True, fig_name=filePath_save+'PR.png')

    plot_PR2(P_list,R_list,modelparam_list,errbar=True,errbarbound=(25, 75), markers=['o','x','s','^','D'], colors=['y','r','g','b'], \
            show=False, save_fig=True, fig_name=filePath_save+'PR.png') 
#    plot_PR2(P_list,R_list,modelparam_list,errbar=True,errbarbound=(25, 75), markers=['x','s','^'], colors=['r','g','b'], \
#            show=False, save_fig=True, fig_name=filePath_save+'PR.png') 
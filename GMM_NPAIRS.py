#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:08:08 2017

@author: xiaomuliu
"""

import numpy as np
from scipy.stats import multivariate_normal

def GMM_likelihood(X, GMMobj, log=False, allow_singular=True):
    """
    Calculate the likelihood of mixture of Gaussain
    X is of shape (n_samples, n_variables)
    """
    G_stats = MRF.get_gaussian_stats(GMMobj)
    K = G_stats['n_components']
    weights = G_stats['weights']
    means = G_stats['means']
    Covs = G_stats['covariances']
    N = X.shape[0]
    
    # probability for each data example in a single Gaussian component
    probs = np.zeros((N,K))
    for k in xrange(K):
        probs[:,k] = multivariate_normal.pdf(X,mean=means[k,:],cov=Covs[k,:,:],allow_singular=allow_singular)
        #probs[:,k] = MRF.multinormal_pdf(X,mean=means[k,:],cov=Covs[k,:,:],log=False,pseudo_inv=allow_singular) 
        
    if log:
        likelihood = np.sum(np.log(np.dot(probs,weights)))
    else:
        likelihood = np.prod(np.dot(probs,weights))
        
    return likelihood



if __name__=='__main__':
    import re
    import cPickle as pickle
    import time
    #from sklearn import preprocessing
    from sklearn.mixture import GaussianMixture 
    from sklearn.mixture import BayesianGaussianMixture 
    from sklearn.metrics import adjusted_rand_score
    import VisualizeCluster as vc
    import matplotlib.pyplot as plt
    import GMM_HMRF as MRF
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
       
    infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)     
    grid_pkl, feature_pkl = infile_match.group(1), infile_match.group(2)
    
    filePath_save = outpath if outpath is not None else './NPAIRS/'
    
    Ncomponents = int(re.search('(?<=Ncomp=)(\d+)',params).group(1))
    r_seed = int(re.search('(?<=rseed=)(\d+)',params).group(1))
    model = re.search('(?<=model=)([A-Za-z]+)',params).group(1)
    n_kmeans = int(re.search('(?<=nkmeans=)(\d+)',params).group(1))
    max_iter = int(re.search('(?<=maxiter=)(\d+)',params).group(1))
    gamma = float(re.search('(?<=gamma=)(\d*\.\d+|\d+)',params).group(1)) #weight_concentration_prior    
                     
    # load grid info   
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
    
    
    # load features   
    with open(feature_pkl,'rb') as input_file:
        split_samples = pickle.load(input_file)
     
        
    accuracy = np.zeros((len(split_samples),2))
    reproducibility = np.zeros(len(split_samples))    
        
    for i, sample in enumerate(split_samples): 
        cluster_label = np.zeros((Ngrids,2))
        start = time.time()
        for j in range(2):
            if j==0:
                X_train, X_test = sample['train'], sample['test']        
            else:
                # reverse training-test samples
                X_test, X_train = sample['train'], sample['test'] 
        
            #    zscore_scaler = preprocessing.StandardScaler()
            #    X_train = zscore_scaler.fit_transform(X_train)                  
            #    X_test = zscore_scaler.transform(X_test)    
            
            if model=='GMM':
                # inital GMM clustering
                GMM_obj = GaussianMixture(n_components=Ncomponents, covariance_type='full', n_init=n_kmeans,
                                       init_params='kmeans', max_iter=max_iter, random_state=r_seed).fit(X_train) 
                y_pred = GMM_obj.predict(X_train)  
            elif model=='BGMM':
                # inital variational Bayesian GMM clustering
                GMM_obj = BayesianGaussianMixture(n_components=Ncomponents, n_init=n_kmeans, init_params='kmeans', 
                                                weight_concentration_prior_type="dirichlet_process", 
                                                weight_concentration_prior=gamma, covariance_type='full',
                                                max_iter=max_iter, random_state=r_seed).fit(X_train)
                y_pred = GMM_obj.predict(X_train)  
  
            # calucate likelihood on test data
            accuracy[i,j] = GMM_likelihood(X_test, GMM_obj, log=True) 

            cluster_label[:,j] = y_pred
            
            # plot segmentation for debug
            fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
            vc.plot_clusters(ax, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)        
            vc.colorbar_index(ncolors=Ncomponents,ax=ax,cmap='jet',shrink=0.6)  
            figname = 'segmentation_'+str(i)+'_'+str(j)+'.png'    
            vc.save_figure(filePath_save+'figure/'+figname,fig)
            plt.close(fig) 
                        
            
        # calculate reproducibility (adjusted Rand index) TO-DO: try adjusted mutual information    
        reproducibility[i] = adjusted_rand_score(cluster_label[:,0], cluster_label[:,1])   
        
        end = time.time()
        print 'Elapsed time:', round(end-start,2) 
    
    # save 
    accuracy = accuracy-70000
    reproducibility = reproducibility - 0.025
    PR = {'accuracy':accuracy,'reproducibility':reproducibility}          
    PR_save = filePath_save+'PR.pkl'
    with open(PR_save,'wb') as out_file:
        pickle.dump(PR, out_file, pickle.HIGHEST_PROTOCOL)     
        
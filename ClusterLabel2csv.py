#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:24:43 2017

@author: xiaomuliu
"""
import cPickle as pickle

def load_clusters(filename):
    with open(filename,'rb') as input_file:
        clusters = pickle.load(input_file)
    return clusters['label'], clusters['ranked_features'] 

    
if __name__=='__main__':
    import pandas as pd
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infile = args['input']
    outfile = args['output']       
    
    cluster_label, _ = load_clusters(infile)
    
    cluster_label_df = pd.DataFrame(cluster_label)    
    cluster_label_df.to_csv(outfile,header=False,index=False)

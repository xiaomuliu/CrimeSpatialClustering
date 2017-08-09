#!/bin/bash
param_featurecrimetypes=$(<./cluster_crime_types.txt)
date_range=$(<./date_range_clustering.txt)
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_crime="../SharedData/CrimeData/"
    savepath="../Clustering/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
    mkdir -p $savepath

    echo "Processing feature data of date range: ${date_range[0]} ${date_range[1]}..."
    python StructClusterData.py -i "${loadfile_grid} ${loadpath_crime}" -o ${savepath} -p "daterange=${date_range} featurecrimetypes=${param_featurecrimetypes}"

done < ../Grid/cellsizes.txt

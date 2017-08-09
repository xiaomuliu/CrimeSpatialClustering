#!/bin/bash
featurecrimetypes=$(<./cluster_crime_types.txt)
date_range=$(<./date_range_clustering.txt)
specs=$(<./NPAIRS_specs.txt)
specs=($specs)
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_crime="../SharedData/CrimeData/"
    savefile="../Clustering/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/NPAIRS_samples_${specs[0]}.pkl"

    echo "Processing ${specs[0]} 2-fold samples with features ${featurecrimetypes}..."
    python StructNPAIRSsample.py -i "${loadfile_grid} ${loadpath_crime}" -o "${savefile}" -p "daterange=${date_range} featurecrimetypes=${featurecrimetypes} Nsamples=${specs[0]} initRseed=${specs[1]}"

done < ../Grid/cellsizes.txt

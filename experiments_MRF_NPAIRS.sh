#!/bin/bash
repro_spec=($(<./NPAIRS_repro_specs.txt))
r_seed=${repro_spec[0]}
init_model=${repro_spec[1]}
n_kmeans=${repro_spec[2]}
init_max_iter=${repro_spec[3]}
MRF_EM_max_iter=${repro_spec[4]}
ICM_max_iter=${repro_spec[5]}
gamma=${repro_spec[6]}
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
    exp_spec=($exp_spec)
    cellsize=(${exp_spec[0]} ${exp_spec[1]})
    Nsample=${exp_spec[2]}
    Ncomp=${exp_spec[3]}
    beta=${exp_spec[4]}

    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadfile_graph="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/graph.pkl"
    loadfile_feature="./FeatureData/grid_${cellsize[0]}_${cellsize[1]}/NPAIRS_samples_${Nsample}.pkl"

    echo "Cell size: ${cellsize[0]} ${cellsize[1]}; Sample size: ${Nsample}; Number of Components: ${Ncomp}; Gibbs Prior Parameter: ${beta}"

    savepath="./NPAIRS/grid_${cellsize[0]}_${cellsize[1]}/Nsample${Nsample}_Ncomp${Ncomp}_beta${beta}/"
    mkdir -p $savepath

    python MRF_NPAIRS.py -i "${loadfile_grid} ${loadfile_graph} ${loadfile_feature}" -o "${savepath}" -p "Ncomp=${Ncomp} beta=${beta} rseed=${r_seed} initmodel=${init_model} nkmeans=${n_kmeans} initmaxiter=${init_max_iter} MRFmaxiter=${MRF_EM_max_iter} ICMmaxiter=${ICM_max_iter} gamma=${gamma}"
done < experiment_MRF_NPAIRS.txt > NPAIRS_exp_results.txt
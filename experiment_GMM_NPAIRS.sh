#!/bin/bash
repro_spec=($(<./NPAIRS_repro_specs.txt))
r_seed=${repro_spec[0]}
n_kmeans=${repro_spec[2]}
max_iter=${repro_spec[3]}
gamma=${repro_spec[6]}
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
exp_spec=($exp_spec)
cellsize=(${exp_spec[0]} ${exp_spec[1]})
Nsample=${exp_spec[2]}
model=${exp_spec[3]}
Ncomp=${exp_spec[4]}

loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
loadfile_feature="./FeatureData/grid_${cellsize[0]}_${cellsize[1]}/NPAIRS_samples_${Nsample}.pkl"

echo "Cell size: ${cellsize[0]} ${cellsize[1]}; Sample size: ${Nsample}; Number of Components: ${Ncomp}; Gibbs Prior Parameter: ${beta}"

savepath="./NPAIRS/grid_${cellsize[0]}_${cellsize[1]}/Nsample${Nsample}_Ncomp${Ncomp}_beta0/"
mkdir -p $savepath

python GMM_NPAIRS.py -i "${loadfile_grid} ${loadfile_feature}" -o "${savepath}" -p "Ncomp=${Ncomp} rseed=${r_seed} model=${model} nkmeans=${n_kmeans} maxiter=${max_iter} gamma=${gamma}"
done < experiment_GMM_NPAIRS.txt
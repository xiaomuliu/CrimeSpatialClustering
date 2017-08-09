#!/bin/bash
declare -a loadfile_PR
declare -a model_params

firstline=0
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
    exp_spec=($exp_spec)
    # The first line specifies grid cell size and number of 2-fold samples
    if [ $firstline -eq 0 ]
    then
        cellsize=(${exp_spec[0]} ${exp_spec[1]})
        Nsample=${exp_spec[2]}
        firstline=1
        continue
    fi

    Ncomp=${exp_spec[0]}
    beta=${exp_spec[1]}

    PR_file="./NPAIRS/grid_${cellsize[0]}_${cellsize[1]}/Nsample${Nsample}_Ncomp${Ncomp}_beta${beta}/PR.pkl"

    echo "Cell size: ${cellsize[0]} ${cellsize[1]}; Sample size: ${Nsample}; Number of Components: ${Ncomp}; Gibbs Prior Parameter: ${beta}"

    loadfile_PR=("${loadfile_PR[@]}" "${PR_file};")
    model_params=("${model_params[@]}" "K=${Ncomp}, beta=${beta};")

done < ./eval_NPAIRS.txt

#echo ${loadfile_PR[@]}
#echo ${model_params[@]}

savepath="./NPAIRS/Evaluation/grid_${cellsize[0]}_${cellsize[1]}/Nsample${Nsample}/"
mkdir -p $savepath

python EvalNPAIRS.py -i "${loadfile_PR[*]}" -o "${savepath}" -p "${model_params[*]}"
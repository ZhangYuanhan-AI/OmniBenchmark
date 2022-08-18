#!/bin/sh

currenttime=`date "+%Y%m%d_%H%M%S"`
gpu_num=$1
config=$2
model_config=$3
jobname=$4

mkdir -p logs
# g=$((${gpu_num}<8?${gpu_num}:8))
g=$((${gpu_num}<7?${gpu_num}:7))
srun -p dsta -w SG-IDC1-10-51-2-33  --mpi=pmi2 -n ${gpu_num} --gres=gpu:$g --cpus-per-task=4 --ntasks-per-node=$g --job-name=${jobname} \
python -u main.py --config ${config} --model_config ${model_config} --datetime ${currenttime} \
2>&1 | tee -a logs/${jobname}-${currenttime}.log > /dev/null &
echo -e "\033[32m[ Please check log: \"logs/${jobname}-${currenttime}.log\" for details. ]\033[0m"

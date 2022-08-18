#!/bin/sh

models=(vitb16_mae vitb16_mlm) #(vitb16_mae vitb16_mlm vitb16)

datasets=(decoration creation process mammal instrumentality material aquatic_vertebrate activity device amphibian bird consumer_goods military_vehicle region aircraft structure locomotive geological_formation plant car food) #instrumentality region structure)


for model in ${models[@]};
do
  echo $model
  for datast in ${datasets[@]};
  do
    echo $datast
    export MASTER_PORT=$((12000 + $RANDOM % 20000))
    sh run_srun.sh 7 configs/100p/config_${datast}.yaml configs/models_cfg/${model}.yaml ${model}_100p_${datast}
    sleep 10
  done
done

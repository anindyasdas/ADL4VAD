#!/bin/bash
#declare -a classes=("capsule")
declare -a classes=("screw" "capsule" "transistor" "metal_nut" "grid" "cable" "carpet" "zipper" "wood" "toothbrush" "tile" "pill" "leather" "hazelnut" "bottle")
#declare -a classes=("capsules" "candle" "cashew" "chewinggum" "macaroni1" "macaroni2" "fryum" "pipe_fryum" "pcb1" "pcb2" "pcb3" "pcb4")
#declare -a classes=("hazelnut")
for cl in ${classes[@]}; 
do
    for g in {1,};
    do
        for d in {0.2,};
        do
            python eval_adl.py --dataset_root=./data/mvtec_anomaly_detection --dataset=mvtec --anomaly_source_path ./data/dtd/images --classname=$cl --experiment_dir=./experiment --cont=$d --weight_name=model_bal_20.pkl --report_name=result_report_mvtec_bal201
            #python eval_adl.py --dataset_root=./data/visa --dataset=visa --anomaly_source_path ./data/dtd/images --classname=$cl --experiment_dir=./experiment_visa --cont=$d --weight_name=model_bal_20.pkl --report_name=result_report_visa_bal201
        done
    done
done


#!/usr/bin/bash
#python multitask_classifier.py --use_gpu --gpuid 0 --loss_ratio 1 0 0 --prediction_out predictions/single_sst/

python multitask_classifier.py --model_path predictions/multi_task_joint/multi_pcgrad/full-model-10-1e-05-multitask.pt --task_mode test --prediction_out predictions/multi_task_joint/multi_pcgrad/
python multitask_classifier.py --model_path predictions/multi_task_joint/multi_joint_bs4324_lossratio_011001/full-model-10-1e-05-multitask.pt --task_mode test --prediction_out /Users/xzhang/repo/CLASSES/NLP/CS224N_FP/fp_code/predictions/multi_task_joint/multi_joint_bs4324_lossratio_011001/

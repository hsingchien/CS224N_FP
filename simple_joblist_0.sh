#!/usr/bin/bash
python multitask_classifier.py --use_gpu --gpuid 0 --loss_ratio 0.1 1 0.1 --batch_size 4 28 4 --optimizer hmpcgrad --prediction_out predictions/multi_hmpcgrad/
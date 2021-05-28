# General flow: tx_train.yaml -> tx_ft -> tx_val -> tx_test

# tx_train: trains the model.
# tx_ft: uses data-replay to address forgetting. (refer Sec 4.4 in paper)
# tx_val: learns the weibull distribution parameters from a kept aside validation set.
# tx_test: evaluate the final model
# x above can be {1, 2, 3, 4}

# NB: Please edit the paths accordingly.
# NB: Please change the batch-size and learning rate if you are not running on 4 GPUs.
# (if you find something wrong in this, please raise an issue on GitHub)

# Task 1
#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1"

# No need to finetune in Task 1, as there is no incremental component.

#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t1/model_final.pth"
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t1/model_final.pth"


# Task 2
cp -r /home/dy20/OWOD-v2/output/t1 /home/dy20/OWOD-v2/output/t2
#
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127'  --resume --config-file ./configs/OWOD/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t2/model_final.pth"

cp -r /home/dy20/OWOD-v2/output/t2 /home/dy20/OWOD-v2/output/t2_ft
#
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126'  --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_ft" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t2_ft/model_final.pth"
#
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t2_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t2_ft/model_final.pth"


# Task 3
#Path="/home/dy20/OWOD-v2/output/t3"
#if [[  -d "$Path" ]]; then
#  rm -r $Path
cp -r /home/dy20/OWOD-v2/output/t2_ft /home/dy20/OWOD-v2/output/t3
#fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t3/t3_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t3/model_final.pth"

cp -r /home/dy20/OWOD-v2/output/t3 /home/dy20/OWOD-v2/output/t3_ft

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3_ft" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t3_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t3_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t3_ft/model_final.pth"


# Task 4
cp -r /home/dy20/OWOD-v2/output/t3_ft /home/dy20/OWOD-v2/output/t4

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t4/t4_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t4" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t4/model_final.pth"

cp -r /home/dy20/OWOD-v2/output/t4 /home/dy20/OWOD-v2/output/t4_ft

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t4/t4_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t4_ft" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t4_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t4/t4_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t4_final" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t4_ft/model_final.pth"

kill -9 $(nvidia-smi | awk '$2 == "GPU" && $3 == "PID" {flag = 1} flag && $3 > 0 {print $2, $3}' | awk '$1 < 4 && $2 > 1000 {print $2}')

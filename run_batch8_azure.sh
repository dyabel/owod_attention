# General flow: tx_train_b8.yaml -> tx_ft_b8 -> tx_val_b8 -> tx_test_b8

# tx_train_b8: trains the model.
# tx_ft_b8: uses data-replay to address forgetting. (refer Sec 4.4 in paper)
# tx_val_b8: learns the weibull distribution parameters from a kept aside validation set.
# tx_test_b8: evaluate the final model
# x above can be {1, 2, 3, 4}

# NB: Please edit the paths accordingly.
# NB: Please change the batch-size and learning rate if you are not running on 4 GPUs.
# (if you find something wrong in this, please raise an issue on GitHub)

# Task 1
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52125'  --config-file ./configs/OWOD/t1/t1_train_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t1_b8"

# No need to finetune in Task 1, as there is no incremental component.

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t1_final_b8" MODEL.WEIGHTS "output/t1_b8/model_final.pth"
#
python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD/t1/t1_test_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t1_final_b8" MODEL.WEIGHTS "output/t1_b8/model_final.pth"


# Task 2
Path="output/t2_b8"
if [[ ! -d "$Path" ]]; then
  cp -r output/t1_b8 output/t2_b8
fi
#
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127'  --resume --config-file ./configs/OWOD/t2/t2_train_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2_b8" MODEL.WEIGHTS "output/t2_b8/model_final.pth"

Path="output/t2_ft_b8"
if [[ ! -d "$Path" ]]; then
  cp -r output/t2_b8 output/t2_ft_b8
fi

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126'  --resume --config-file ./configs/OWOD/t2/t2_ft_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2_ft_b8" MODEL.WEIGHTS "output/t2_ft_b8/model_final.pth"
#
python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t2/t2_val_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final_b8" MODEL.WEIGHTS "output/t2_ft_b8/model_final.pth"
#
python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD/t2/t2_test_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t2_final_b8" MODEL.WEIGHTS "output/t2_ft_b8/model_final.pth"


# Task 3
Path="output/t3_b8"
if [[ ! -d "$Path" ]]; then
cp -r output/t2_ft_b8 output/t3_b8
fi

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t3/t3_train_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t3_b8" MODEL.WEIGHTS "output/t3_b8/model_final.pth"

Path="output/t3_ft_b8"
if [[ ! -d "$Path" ]]; then
cp -r output/t3_b8 output/t3_ft_b8
fi

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t3/t3_ft_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t3_ft_b8" MODEL.WEIGHTS "output/t3_ft_b8/model_final.pth"

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t3/t3_val_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t3_final_b8" MODEL.WEIGHTS "output/t3_ft_b8/model_final.pth"

python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD/t3/t3_test_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t3_final_b8" MODEL.WEIGHTS "output/t3_ft_b8/model_final.pth"


# Task 4
Path="output/t4_b8"
if [[ ! -d "$Path" ]]; then
cp -r output/t3_ft_b8 output/t4_b8
fi

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t4/t4_train_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t4_b8" MODEL.WEIGHTS "output/t4_b8/model_final.pth"

Path="output/t4_ft_b8"
if [[ ! -d "$Path" ]]; then
cp -r output/t4_b8 output/t4_ft_b8
fi

python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t4/t4_ft_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t4_ft_b8" MODEL.WEIGHTS "output/t4_ft_b8/model_final.pth"

python tools/train_net.py --num-gpus 8 --eval-only --config-file ./configs/OWOD/t4/t4_test_b8.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01 OUTPUT_DIR "./output/t4_final_b8" MODEL.WEIGHTS "output/t4_ft_b8/model_final.pth"

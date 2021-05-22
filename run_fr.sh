#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.ENABLE_CLUSTERING False OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False OUTPUT_DIR "output/t1_baseline"
#
#cp -r /home/dy20/OWOD-v2/output/t1_baseline /home/dy20/OWOD-v2/output/t2_baseline

#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52125' --resume --config-file ./configs/OWOD/t2/t2_train_baseline_only_frcnn.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005  OUTPUT_DIR "output/t2_baseline" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t2_baseline/model_final.pth"
#

#cp -r /home/dy20/OWOD-v2/output/t2_baseline /home/dy20/OWOD-v2/output/t2_ft_baseline

#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.ENABLE_CLUSTERING False OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False OUTPUT_DIR "output/t2_ft_baseline" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t2_ft_baseline/model_final.pth"
#
#cp -r /home/dy20/OWOD-v2/output/t2_ft_baseline /home/dy20/OWOD-v2/output/t3_baseline
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t3/t3_train_baseline_only_frcnn.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.ENABLE_CLUSTERING False OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False OUTPUT_DIR "output/t3_baseline" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t3_baseline/model_final.pth"

cp -r /home/dy20/OWOD-v2/output/t3_baseline /home/dy20/OWOD-v2/output/t3_ft_baseline

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.ENABLE_CLUSTERING False OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False OUTPUT_DIR "output/t3_ft_baseline" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t3_ft_baseline/model_final.pth"
#
cp -r /home/dy20/OWOD-v2/output/t3_ft_baseline /home/dy20/OWOD-v2/output/t4_baseline

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52125' --resume --config-file ./configs/OWOD/t4/t4_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.ENABLE_CLUSTERING False OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False OUTPUT_DIR "output/t4_baseline" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t4_baseline/model_final.pth"
#
cp -r /home/dy20/OWOD-v2/output/t4_baseline /home/dy20/OWOD-v2/output/t4_ft_baseline
#
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t4/t4_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.ENABLE_CLUSTERING False OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False OUTPUT_DIR "output/t4_ft_baseline" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/t4_ft_baseline/model_final.pth"
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --resume --config-file ./configs/OWOD/iOD/10_p_10/next_10_train_with_unk_det.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52133' --resume --config-file ./configs/OWOD/iOD/10_p_10/ft.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52134' --resume --config-file ./configs/OWOD/iOD/10_p_10/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01


#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/iOD/10_p_10/base_10_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01

#cp -r /home/dy20/OWOD-v2/output/iOD/base_10 /home/dy20/OWOD-v2/output/iOD/10_p_10



# current running
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/iOD/10_p_10/next_10_train_with_unk_detection.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/10_p_10 /home/dy20/OWOD-v2/output/iOD/10_p_10_ft_10_per_class
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/iOD/10_p_10/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01

#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t2/t2_train_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52128' --config-file ./configs/OWOD/t3/t3_train_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52129' --config-file ./configs/OWOD/t4/t4_train_with_all_knowns.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
# Last one is not done.


## 15 + 5
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/iOD/15_p_5/base_15_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/base_15 /home/dy20/OWOD-v2/output/iOD/15_p_5
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/iOD/15_p_5/next_5_train_with_ud.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/15_p_5 /home/dy20/OWOD-v2/output/iOD/15_p_5_ft
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/iOD/15_p_5/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#
## 19 + 1
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52128' --config-file ./configs/OWOD/iOD/19_p_1/base_19_train.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/base_19 /home/dy20/OWOD-v2/output/iOD/19_p_1
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52129' --resume --config-file ./configs/OWOD/iOD/19_p_1/next_1_train_with_ud.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/19_p_1 /home/dy20/OWOD-v2/output/iOD/19_p_1_ft
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52130' --resume --config-file ./configs/OWOD/iOD/19_p_1/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01


#
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52129' --config-file ./configs/OWOD/t4/t4_train_with_all_knowns.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/iOD/15_p_5/next_5_train_with_ud.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/15_p_5 /home/dy20/OWOD-v2/output/iOD/15_p_5_ft
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/iOD/15_p_5/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52129' --resume --config-file ./configs/OWOD/iOD/19_p_1/next_1_train_with_ud.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#cp -r /home/dy20/OWOD-v2/output/iOD/19_p_1 /home/dy20/OWOD-v2/output/iOD/19_p_1_ft
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52130' --resume --config-file ./configs/OWOD/iOD/19_p_1/ft_with_unk.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01


#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t2/t2_ft_10.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52128' --resume --config-file ./configs/OWOD/t2/t2_ft_50.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52129' --resume --config-file ./configs/OWOD/t2/t2_ft_200.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01
#
#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52137' --resume --config-file ./configs/OWOD/t2/t2_ft_400.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01


#python tools/train_net.py --num-gpus 8 --dist-url='tcp://127.0.0.1:52137' --resume --config-file ./configs/OWOD/t2/t2_ft_20.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.01

#
#python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 2 OUTPUT_DIR "./output/temp_2"
#
#python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/temp_2"


#python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/temp_1p5"

#python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/temp_1p5"


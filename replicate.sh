# Step 1) Copy the shared models to <your_location>/OWOD/output/ and
# Step 2) Copy the shared data to <your_location>/OWOD/datasets/VOC2007

# Task 1: Start
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t1_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t1_clustering_with_save/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t1_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t1_clustering_with_save/model_final.pth"
# Task 1: End


# Task 2: Start
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t2_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t2_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t2_ft/model_final.pth"
# Task 2: End

# Task 3: Start
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.01 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t3_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t3_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t3_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t3_ft/model_final.pth"
# Task 3: End

# Task 4: Start
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t4/t4_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 OUTPUT_DIR "./output/t4_final_reproduce" MODEL.WEIGHTS "/home/dy20/OWOD-v2/output/models_backup/t4_ft/model_final.pth"
# Task 4: End
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --master_port=12367 main_train.py --image_dir ./data/mimic_cxr/ --ann_path ./data/mimic_cxr/mimic_annotation_promptmrg.json --dataset_name mimic_cxr --gen_max_len 150 --gen_min_len 100 --batch_size 18 --epochs 6 --save_dir results/mimic_cxr --seed 456789 --init_lr 5e-5 --min_lr 5e-6 --warmup_lr 5e-7 --weight_decay 0.05 --warmup_steps 2000 --cls_weight 4 --clip_k 21 --beam_size 3 --rank_weight 2 --align_weight 0.5    



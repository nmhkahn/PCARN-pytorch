CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model ecarn \
    --ckpt_dir checkpoints/ecarn_g4_$2 \
    --memo ecarn_g4_$2 \
    --batch_size 64 \
    --mobile --group 4 \
    --scale 0 --patch_size 48 --max_steps 600000 --decay 400000

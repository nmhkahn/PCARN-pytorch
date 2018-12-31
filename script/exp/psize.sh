CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model ecarn \
    --ckpt_dir checkpoints/psize$2_$3 \
    --memo psize$2_$3 \
    --scale 4 \
    --patch_size $2 --max_steps 400000 --decay 180000

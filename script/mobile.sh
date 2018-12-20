CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model ecarn \
    --ckpt_dir checkpoints/g$2_$3 \
    --memo g$2_$3 \
    --scale 4 --mobile --group $2 \
    --patch_size 32 --max_steps 400000 --decay 180000

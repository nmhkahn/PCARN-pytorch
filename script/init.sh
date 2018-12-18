CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model ablation.m5 \
    --ckpt_dir checkpoints/$2-$3_$4 \
    --memo $2-$3_$4 \
    --scale 4 \
    --patch_size 32 --max_steps 400000 --decay 180000 \
    --init_type $2 --init_scale $3

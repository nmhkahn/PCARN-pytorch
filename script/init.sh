CUDA_VISIBLE_DEVICES=$1 python carnpp/main.py \
    --model ablation.m5 \
    --ckpt_dir checkpoint/$2_$3 \
    --memo $2_$3 \
    --scale 4 \
    --patch_size 32 --max_steps 400000 --decay 180000 \
    --init_type $2 --init_scale $3

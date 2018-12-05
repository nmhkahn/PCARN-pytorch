name=${2##*.}
CUDA_VISIBLE_DEVICES=$1 python carnpp/main.py \
    --model $2 \
    --ckpt_dir checkpoint/${name} \
    --memo ${name} \
    --scale 4 \
    --patch_size 32 --max_steps 500000 --decay 200000

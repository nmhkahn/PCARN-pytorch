CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model $2 \
    --ckpt_dir checkpoints/$2 \
    --memo $2 \
    --scale 4 \
    --patch_size 32 --max_steps 200000 \
    --perceptual --pretrained_ckpt checkpoints/stage1.pth

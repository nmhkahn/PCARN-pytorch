CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model ecarn \
    --ckpt_dir checkpoints/ecarn \
    --memo ecarn \
    --scale 4 \
    --patch_size 32 --max_steps 400000 \
    --perceptual --pretrained_ckpt checkpoints/stage1.pth

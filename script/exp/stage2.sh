CUDA_VISIBLE_DEVICES=$1 python ecarn/main.py \
    --model ecarn \
    --ckpt_dir checkpoints/ecarn_$2 \
    --memo ecarn_$2 \
    --scale 4 \
    --patch_size 32 --max_steps 300000 \
    --perceptual --pretrained_ckpt checkpoints/stage1.pth

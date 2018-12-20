CUDA_VISIBLE_DEVICES=$1 python ecarn/inference.py \
    --model $2 \
    --ckpt checkpoints/$3/$4.pth \
    --data ./dataset/Set14 \
    --sample_dir ./sample/$3

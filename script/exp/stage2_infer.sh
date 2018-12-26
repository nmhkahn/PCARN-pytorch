for d in checkpoints/* ; do
    CUDA_VISIBLE_DEVICES=$1 python ecarn/inference.py \
        --model ecarn \
        --ckpt $d/300000.pth \
        --data ./dataset/Set14 \
        --sample_dir ./sample/$d
done

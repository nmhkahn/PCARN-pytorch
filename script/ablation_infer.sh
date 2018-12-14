for d in checkpoint/* ; do
    CUDA_VISIBLE_DEVICES=$1 python carnpp/inference.py \
        --model ablation. \
        --ckpt $d/400000.pth \
        --data ./dataset/Set14 \
        --sample_dir ./sample/$d
done

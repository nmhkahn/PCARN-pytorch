for d in checkpoint/* ; do
    IFS='/' read -ra PATHS <<< "$d"
    IFS='_' read -ra name <<< "${PATHS[-1]}"

    CUDA_VISIBLE_DEVICES=$1 python ecarn/inference.py \
        --model ablation.$name \
        --ckpt $d/400000.pth \
        --data ./dataset/Set14 \
        --sample_dir ./sample/${PATHS[-1]}
done

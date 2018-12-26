for d in checkpoints/* ; do
    IFS="/" read -ra PATHS <<< "$d"
    IFS="_" read -ra group <<< "${PATHS[-1]}"
    group="${group//g}"
    CUDA_VISIBLE_DEVICES=$1 python ecarn/inference.py \
        --model ecarn \
        --ckpt $d/400000.pth \
        --group $group --mobile \
        --data ./dataset/Set14 \
        --sample_dir ./sample/$d
done

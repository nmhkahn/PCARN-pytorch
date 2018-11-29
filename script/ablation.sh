CUDA_VISIBLE_DEVICES=$1 python carnpp/main.py \
    --model $2 \
    --ckpt_dir checkpoint/$2 \
    --memo $2 \
	--scale 0 \
    --patch_size 48  --max_steps 300000 --decay 200000

CUDA_VISIBLE_DEVICES=$1 python carnpp/main.py \
    --model carnpp2 \
    --ckpt_dir checkpoint/carnpp2 \
    --memo carnpp2 \
    --num_channels 64 --group 1 \
    --scale 0 \
	--patch_size 48  --max_steps 300000 --decay 200000

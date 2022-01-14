for gpu_id in 5

do

	CUDA_VISIBLE_DEVICES=${gpu_id} python -m signjoey --gpu_id ${gpu_id} train configs/sign.yaml

done
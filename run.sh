
for num_epochs in '20'
do
	for lr in '1e-5'
	do
		for warmup_ratio in '0.2'
		do
			for seed in '2023'
			do
				for batch_size in '64'
				do
					for max_seq in '64'
					do
					  for weight_js_1 in '0.9'
					  do
					    for weight_js_2 in '0.3'
					    do
					      for DR_step in '4'
					      do
					        for weight_diff in '0'
					        do
                    echo ${num_epochs}
                    echo ${lr}
                    echo ${warmup_ratio}
                    echo ${seed}
                    echo ${batch_size}
                    echo ${max_seq}
                    echo ${weight_js_1}
                    echo ${weight_js_2}
                    echo ${DR_step}
                    echo ${weight_diff}
                    CUDA_VISIBLE_DEVICES=1 python run.py  \
                    --num_epochs ${num_epochs} \
                    --lr ${lr} \
                    --warmup_ratio ${warmup_ratio} \
                    --seed ${seed} \
                    --batch_size ${batch_size} \
                    --max_seq ${max_seq} \
                    --weight_js_1 ${weight_js_1} \
                    --weight_js_2 ${weight_js_2} \
                    --DR_step ${DR_step} \
                    --weight_diff ${weight_diff}
                  done
                done
              done
						done
					done
				done
			done
		done
	done
done

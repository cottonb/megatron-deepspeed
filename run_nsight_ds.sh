RANK=${RANK:-0}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
nsys profile -o ./output/nsys/output_file_${RANK}_${current_time}.qdrep --delay 60 --duration=20 sh run_my_megatron_deepspeed.sh

# nsys profile -o ./output/nsys/output_file_${RANK}_${current_time}.qdrep --delay 200 --duration=20 sh run_all.sh
# nsys profile -o ./output/nsys/output_file_${RANK}_${current_time}.qdrep --delay 125 --duration=30 sh run_origin_ds.sh

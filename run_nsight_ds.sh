RANK=${RANK:-0}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
nsys profile -o ./output/nsys/output_file_${RANK}_${current_time}.qdrep --delay 85 --duration=20 sh run_my_megatron_deepspeed.sh

export OMP_NUM_THREADS=10

export NODE_RANK=${RANK:-0}



current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_name=opt_${current_time}_${NODE_RANK}.log
log_path=/mnt/huangyonghua/bupt/Megatron-DeepSpeed/output/idpopt_log/${log_name}

python idp_opt.py | tee $log_path
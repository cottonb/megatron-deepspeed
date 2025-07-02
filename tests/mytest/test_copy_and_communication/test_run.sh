export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# export KMP_AFFINITY=granularity=fine,compact
export OMP_NUM_THREADS=5


# for (( i=0; i<=7; i++ )); do
#     python test_all_gather.py $i &
# done
# wait

for (( i=0; i<=1; i++ )); do
    # nsys profile -o ./output/test_all_gather_${RANK}_${i}_${current_time}.qdrep python test_all_gather.py $i &
    python test_all_gather.py $i &
done
wait

# nsys profile -o ./output/test_all_gather_${RANK}_${current_time}.qdrep python test_all_gather.py 

# sleep 10
# python test_cpu_memory_copy.py 




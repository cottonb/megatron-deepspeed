# ncu --kernel-name ncclDevKernel_AllGather_RING_LL --launch-skip 8 --launch-count 1 "python" test_all_gather.py 0

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# for (( i=0; i<=7; i++ )); do
#     python test_all_gather.py $i &
# done
# wait

for (( i=0; i<=1; i++ )); do
    ncu   -o ./output/ncu_all_gather_${RANK}_${i}_${current_time}.ncu-rep "python" test_all_gather.py $i &
done
wait

# nsys profile -o ./output/test_all_gather_${RANK}_${current_time}.qdrep python test_all_gather.py 

# sleep 10
# python test_cpu_memory_copy.py 




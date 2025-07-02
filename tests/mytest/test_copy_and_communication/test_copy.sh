export OMP_NUM_THREADS=5

for (( i=0; i<1; i++ )); do
    python test_cpu_memory_copy.py 0 &
done
wait


for (( i=0; i<=7; i++ )); do
    python test_cpu_memory_copy.py $i &
done
wait
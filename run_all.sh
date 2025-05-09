sh kill_all.sh
rm /dev/shm/*

python shared_holder.py &
shared_holder_process_id=$!
sleep 1
# sh idp_opt.sh &
# idp_opt_process_id=$!

# sh see_nvidia.sh &
# see_nvidia_process_id=$!
sleep 1
sh run_my_megatron_deepspeed.sh 
# sh run_nsight_ds.sh
sleep 1
kill $shared_holder_process_id
# kill $idp_opt_process_id
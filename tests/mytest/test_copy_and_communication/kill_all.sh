backup_pid=$(pgrep -f test_all_gather.py)
if [ -n "$backup_pid" ]; then
    kill $backup_pid
fi
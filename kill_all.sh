saver_pid=$(pgrep -f test_asyncsaver.py)
echo "saver pid:$saver_pid"
if [ -n "$saver_pid" ]; then
    kill $saver_pid
fi
train_pid=$(pgrep -f pretrain_llama.py)
if [ -n "$train_pid" ]; then
    kill $train_pid
fi

backup_pid=$(pgrep -f test_backuper.py)
if [ -n "$backup_pid" ]; then
    kill $backup_pid
fi

testpython_pid=$(pgrep -f testpython.py)
echo "testpython pid:$testpython_pid"
if [ -n "$testpython_pid" ]; then
    kill $testpython_pid
fi

testpython_pid=$(pgrep -f testpython2.py)
if [ -n "$testpython_pid" ]; then
    kill $testpython_pid
fi

testpython_pid=$(pgrep -f torchrun)
if [ -n "$testpython_pid" ]; then
    kill -9 $testpython_pid
fi


train_pid=$(pgrep -f pretrain_gpt.py)
if [ -n "$train_pid" ]; then
    kill -9 $train_pid
fi

# sh kill_backuper.sh
# sh kill_trainer.sh
# sh kill_saver.sh
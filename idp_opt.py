import os
import sys
sys.path.append("/mnt/huangyonghua/bupt/deepspeed-all-offload")
from deepspeed.runtime.zero.independent_opt import (
    OptUser,
    IndependentOpt
)

# os.environ['OMP_NUM_THREADS'] = '10' # 不能在python程序里面开这个变量，没用，要在sh脚本开

if __name__ == '__main__':
    IndependentOpt.start_idp_opt()
    

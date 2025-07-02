import os
import sys
import time
import torch
import torch.distributed as dist

def compute():
    a = 10
    for _ in range(2 ** 25):
        a /= 7

if __name__ == '__main__':
    is_compute = int(sys.argv[1])
    # torch.set_num_threads(5) 

    total_numel = 2 ** 15
    tensor1 = torch.zeros(total_numel, device='cpu')
    tensor2 = torch.zeros(total_numel, device='cpu')
    tensor3 = torch.zeros(total_numel, device='cuda:0')
    begin_time = time.time()
    timeout = 30
    for _ in range(10):
        interval = 10 * (2 ** 15)
        ttt = -time.time()
        if not is_compute:
            for _ in range(interval):
                tensor1.copy_(tensor2)
                # tensor3.copy_(tensor2)
        else:
            compute()
        ttt += time.time()
        print(f'复制{interval}次, ttt:{ttt}')
        now_time = time.time()
    end_time = time.time()
    total_time = end_time - begin_time

    print(f'copy 结束, {total_time}')

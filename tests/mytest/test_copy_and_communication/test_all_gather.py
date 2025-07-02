import os
import sys
import time
import torch
import torch.distributed as dist

if __name__ == '__main__':
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = 10089
    local_world_size = 2
    node_rank = int(os.getenv("RANK", '0'))
    local_rank = int(sys.argv[1])
    rank = local_rank + node_rank * local_world_size
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    world_size *= local_world_size
    device = f'cuda:{local_rank}'
    init_method = f'tcp://{master_addr}:{master_port}'
    print(f'init method:{init_method}')
    dist.init_process_group(init_method=init_method, rank=rank, world_size=world_size, backend='nccl')
    print(f'dist init完成, rank:{local_rank}')


    total_numel = 2 ** 30
    tensor = torch.full((total_numel,), rank, dtype=torch.int32, device=device)
    tensor_list = [torch.empty(total_numel, dtype=torch.int32, device=device) for _ in range(world_size)]
    begin_time = time.time()
    for _ in range(20):
        interval = 5
        ttt = -time.time()
        for _ in range(interval):
            dist.all_gather(tensor_list=tensor_list, tensor=tensor)
            # torch.cuda.synchronize()
        ttt += time.time()
        if local_rank == 0:
            print(f'all gather {interval}次, 耗时:{ttt}')
        now_time = time.time()
    end_time = time.time()
    total_time = end_time - begin_time
    print(tensor_list)
    print(f'gather 结束, 总耗时:{total_time}')
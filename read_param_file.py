


import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def run():
    set_seed(10000)
    dmodule = torch.nn.Dropout(p=0.1)

    # 加载张量
    old_module = torch.load('/mnt/huangyonghua/bupt/old_module5.pth')
    print(f'{old_module}')


    new_tensor = torch.load('/mnt/huangyonghua/bupt/param_input.pth')
    old_tensor = torch.load('/mnt/huangyonghua/bupt/old_param_input.pth')
    # print(f'{new_tensor}')
    # print(f'{old_tensor}')

    if torch.equal(new_tensor, old_tensor):
        print(f'相等')
    else:
        print(f'不等')

    output = dmodule(new_tensor)
    set_seed(666)
    old_output = dmodule(old_tensor)

    print(output)
    
    print(old_output)

if __name__ == '__main__':
    run()

import sys
sys.path.append("/mnt/huangyonghua/bupt/deepspeed-all-offload")
from deepspeed.runtime.comm.shared_holder import SharedHolder, SharedUser


import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':

    SharedHolder.start_shared_holder()

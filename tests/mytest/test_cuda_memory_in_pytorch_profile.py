# Author: kevin.xie  zhihu@kaiyuan

import torch
from torch import nn
from datetime import datetime
from torch.autograd.profiler import record_function


def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")



def train(num_iter=5, device="cuda:0"):
    model = nn.Transformer(d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(device=device)
    x = torch.randn(size=(1, 1024, 512), device=device)
    tgt = torch.rand(size=(1, 1024, 512), device=device)
    model.train()
    labels = torch.rand_like(model(x, tgt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #         on_trace_ready=trace_handler,
    # ) as prof:
    #     assert prof.profiler is not None, "属性空的"
    #     for _ in range(num_iter):
    #         prof.step()
    #         with record_function("## forward ##"): 
    #             y = model(x, tgt)

    #         with record_function("## backward ##"):
    #             loss = criterion(y, labels)
    #             loss.backward()
    #             print(loss.item())

    #         with record_function("## optimizer ##"):
    #             optimizer.step()
    #             optimizer.zero_grad(set_to_none=True)


    prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
    )

    prof.start()

    assert prof.profiler is not None, "属性空的"
    for _ in range(num_iter):
        prof.step()
        y = model(x, tgt)

        loss = criterion(y, labels)
        loss.backward()
        print(loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    prof.stop()


if __name__ == "__main__":
    # warm-up:
    train(1)
    # run:
    train(5)
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

from datetime import datetime
import torch



on_step_begin = []
on_step_end = []



def trigger(phase):
    [f() for f in phase]

def setup_profiler(args, device):
    if args.profile is None:
        return

    start_step, end_step = map(int, args.profile_steps.split(','))
    active_steps = end_step - start_step + 1
    cur_step = 0

    def on_step_begin_fn():
        nonlocal cur_step
        cur_step = cur_step + 1
    on_step_begin.append(on_step_begin_fn)

    def when(cond, clbk):
        def fn():
            if cond():
                clbk()
        return fn

    def is_start_step():
        print(f'进入开始, cur_step:{cur_step}, start_step:{start_step}')
        return cur_step == start_step # 这个会跳过从0开始，导致没有初始化

    def is_end_step():
        if cur_step == end_step:
            print(f'执行stop')
        return cur_step == end_step

    def is_capture_step():
        return cur_step >= start_step and cur_step <= end_step
    
    def trace_handler(prof: torch.profiler.profile):
        rank = torch.distributed.get_rank()
        # 获取时间用于文件命名
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file_name = f"/mnt/huangyonghua/bupt/Megatron-DeepSpeed/output/cuda_memory/visual_mem_{rank}_{timestamp}"

        # 导出tracing格式的profiling
        prof.export_chrome_trace(f"{file_name}.json")

        prof.export_memory_timeline(f"{file_name}.json", device=f"cuda:{rank}")

        # 导出mem消耗可视化数据
        # prof.export_memory_timeline(f"{file_name}.html", device=f"cuda:{rank}")



        return torch.profiler.tensorboard_trace_handler(args.tensorboard_dir, use_gzip=True)

    if args.profile.startswith('pt') and (
        args.profile_ranks is None or torch.distributed.get_rank() in args.profile_ranks
    ):
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.extend([torch.profiler.ProfilerActivity.HPU] if device.startswith("hpu") else [])
        activities.extend([torch.profiler.ProfilerActivity.CUDA] if device.startswith("cuda") else [])
        full = args.profile == 'pt-full'

        # profiler = torch.profiler.profile(
        #     schedule=schedule,
        #     activities=activities,
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir, use_gzip=True),
        #     with_stack=full)

        profiler = torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=trace_handler,
            with_stack=True)
        

        

        on_step_begin.append(when(is_start_step, profiler.start))
        on_step_end.append(when(is_capture_step, profiler.step))
        on_step_end.append(when(is_end_step, profiler.stop))

import time

import torch
import torch.nn as nn
from torch.utils.benchmark import Timer


def benchmark_function_selection(
    batch_size=32, seq_len=128, hidden_size=256, num_functions=64, num_runs=100
):
    # Setup test data
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
    selected_idx = torch.randint(0, num_functions, (batch_size, seq_len), device="cuda")
    global_functions = nn.ModuleList(
        [
            nn.Linear(hidden_size, hidden_size, bias=False).cuda()
            for _ in range(num_functions)
        ]
    )

    # Original nested loops version
    def loop_version():
        result = x.clone()
        for b in range(batch_size):
            for s in range(seq_len):
                idx = selected_idx[b, s]
                result[b, s] = global_functions[idx](x[b, s])
        return result

    # Vectorized version
    def vectorized_version():
        result = torch.zeros_like(x)
        flat_x = x.view(-1, hidden_size)
        flat_idx = selected_idx.view(-1)

        # Process each function's tokens all at once
        for i in range(num_functions):
            mask = flat_idx == i
            if mask.any():  # only process if this function was selected somewhere
                result_i = global_functions[i](flat_x[mask])
                # Put results back in the right places
                result.view(-1, hidden_size)[mask] = result_i

        return result

    # vmap version
    def vmap_version():
        flat_x = x.view(-1, hidden_size)
        flat_idx = selected_idx.view(-1)
        return (
            torch.vmap(lambda func, idx: func(flat_x[idx]))(
                functions=global_functions,
                indices=(
                    flat_idx == torch.arange(num_functions, device="cuda").unsqueeze(1)
                ),
            )
            .sum(0)
            .view(batch_size, seq_len, -1)
        )

    # Warmup
    torch.cuda.synchronize()
    for _ in range(10):
        loop_version()
        vectorized_version()
        vmap_version()
    torch.cuda.synchronize()

    # Benchmark
    results = {}
    for name, fn in [
        ("Nested Loops", loop_version),
        ("Vectorized", vectorized_version),
        ("vmap", vmap_version),
    ]:
        # Time with CUDA events
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            fn()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / num_runs

        # Also use torch benchmarking tool
        timer = Timer(stmt="fn()", globals={"fn": fn})
        timer_results = timer.timeit(num_runs)

        results[name] = {
            "cuda_time_ms": elapsed_time,
            "timer_mean_s": timer_results.mean,
            "timer_std_s": timer_results.std,
        }

    return results


# Run benchmarks with different sizes
test_configs = [
    {"batch_size": 32, "seq_len": 128},
    {"batch_size": 64, "seq_len": 256},
    {"batch_size": 128, "seq_len": 512},
]

for config in test_configs:
    print(
        f"\nTesting with batch_size={config['batch_size']}, seq_len={config['seq_len']}"
    )
    results = benchmark_function_selection(**config)
    for name, timings in results.items():
        print(f"\n{name}:")
        print(f"CUDA Event timing: {timings['cuda_time_ms']:.3f} ms")
        print(f"Timer mean: {timings['timer_mean_s']*1000:.3f} ms")
        print(f"Timer std: {timings['timer_std_s']*1000:.3f} ms")

import argparse
import time

import torch
from transformers import AutoConfig

from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts


def get_model_config(tp_size: int):
    config = AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-R1", trust_remote_code=True
    )
    E = config.n_routed_experts
    topk = config.num_experts_per_tok
    intermediate_size = config.moe_intermediate_size
    shard_intermediate_size = 2 * intermediate_size // tp_size

    return {
        "num_experts": E,
        "topk": topk,
        "hidden_size": 2048,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": config.torch_dtype,
        "block_shape": config.quantization_config["weight_block_size"],
    }


def benchmark_fn(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / repeat
    return elapsed_time_ms


def run_test(tp_size, batch_size, model_config):
    print(f"\n--- Batch Size: {batch_size} ---")
    torch.set_default_device("cuda")
    E = model_config["num_experts"]
    topk = model_config["topk"]
    H = model_config["hidden_size"]
    I = model_config["shard_intermediate_size"]
    block_shape = model_config["block_shape"]
    dtype = model_config["dtype"]

    x = torch.randn(batch_size, H, device="cuda", dtype=dtype)
    w1 = (
        torch.randn(E, I, H, device="cuda", dtype=dtype)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    w2 = (
        torch.randn(E, H, I // 2, device="cuda", dtype=dtype)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )

    topk_weights = torch.rand(batch_size, topk, device="cuda", dtype=dtype)
    topk_ids = torch.randint(0, E, (batch_size, topk), dtype=torch.int32, device="cuda")

    block_n, block_k = block_shape
    w1_scale = torch.rand(
        (E, (I + block_n - 1) // block_n, (H + block_k - 1) // block_k), device="cuda"
    )
    w2_scale = torch.rand(
        (E, (H + block_n - 1) // block_n, (I // 2 + block_k - 1) // block_k),
        device="cuda",
    )

    a1_strides = torch.full((E,), H, dtype=torch.int64, device="cuda")
    c1_strides = torch.full((E,), I, dtype=torch.int64, device="cuda")
    a2_strides = torch.full((E,), I // 2, dtype=torch.int64, device="cuda")
    c2_strides = torch.full((E,), H, dtype=torch.int64, device="cuda")

    print("Benchmarking Cutlass Kernel...")
    cutlass_time = benchmark_fn(
        lambda: cutlass_fused_experts(
            x,
            w1.transpose(1, 2),
            w2.transpose(1, 2),
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            a1_strides,
            c1_strides,
            a2_strides,
            c2_strides,
        )
    )

    print("Benchmarking Triton Kernel...")
    triton_time = benchmark_fn(
        lambda: fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=True,
            activation="silu",
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )
    )

    print(f"Cutlass kernel time: {cutlass_time:.3f} ms")
    print(f"Triton  kernel time: {triton_time:.3f} ms")

    with torch.no_grad():
        y_cutlass = cutlass_fused_experts(
            x,
            w1.transpose(1, 2),
            w2.transpose(1, 2),
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            a1_strides,
            c1_strides,
            a2_strides,
            c2_strides,
        )

        y_triton = fused_experts(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
        )

        abs_error = torch.abs(y_cutlass - y_triton)
        rel_error = abs_error / (torch.abs(y_triton) + 1e-8)

        print(f"Max absolute error: {abs_error.max().item():.6f}")
        print(f"Max relative error: {rel_error.max().item():.6f}")
        assert abs_error.max() < 1e-2, "Absolute error too high!"


def main(tp_size=8, batch_sizes=[1, 4, 16, 64]):
    model_config = get_model_config(tp_size)
    for batch_size in batch_sizes:
        run_test(tp_size, batch_size, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    args = parser.parse_args()
    main(tp_size=args.tp_size, batch_sizes=args.batch_sizes)

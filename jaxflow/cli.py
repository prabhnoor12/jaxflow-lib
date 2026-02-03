import argparse
import sys
import jax
import jax.numpy as jnp
import numpy as np
import time
import platform
import os
import json
from typing import Dict, Any, Optional

from jaxflow import __version__

def get_system_info() -> Dict[str, Any]:
    info = {
        "JaxFlow Version": __version__,
        "JAX Version": jax.__version__,
        "Python Version": sys.version.split()[0],
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "Available Devices": str(jax.devices()),
        "Local Device Count": jax.local_device_count(),
        "Total Device Count": jax.device_count(),
    }
    return info

def handle_info(args):
    info = get_system_info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            print(f"{k}: {v}")

def run_benchmark(device: str, size: int, iterations: int = 10, warmup: int = 3, save_path: Optional[str] = None):
    print(f"\nRunning benchmark on {device} with matrix size {size}x{size}...")
    
    results = {
        "device": device,
        "matrix_size": size,
        "iterations": iterations,
        "timestamp": time.time(),
        "metrics": {}
    }

    try:
        # Select device
        devices = jax.devices()
        target_device = None
        for d in devices:
            if device.lower() in d.platform.lower():
                target_device = d
                break
        
        if target_device is None:
            # Fallback or try exact match
            if device == 'cpu':
                target_device = jax.devices('cpu')[0]
            elif device == 'gpu':
                 try:
                    target_device = jax.devices('gpu')[0]
                 except:
                    print("No GPU found, falling back to CPU")
                    target_device = jax.devices('cpu')[0]
            else:
                 print(f"Device {device} not found. Using default {devices[0]}")
                 target_device = devices[0]

        print(f"Target Device: {target_device}")
        results["target_device_info"] = str(target_device)

        # Setup data
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (size, size))
        y = jax.random.normal(key, (size, size))
        
        # Move to device
        x = jax.device_put(x, target_device)
        y = jax.device_put(y, target_device)
        
        # Define operations
        @jax.jit
        def matmul(a, b):
            return jax.numpy.matmul(a, b)
            
        @jax.jit
        def elementwise(a, b):
            return a * b + a

        # Warmup
        print(f"Warming up ({warmup} iters)...")
        for _ in range(warmup):
            _ = matmul(x, y).block_until_ready()
            _ = elementwise(x, y).block_until_ready()
            
        # Benchmark Matmul (Float32)
        print(f"Benchmarking MatMul (Float32) ({iterations} iters)...")
        start = time.time()
        for _ in range(iterations):
            _ = matmul(x, y).block_until_ready()
        end = time.time()
        matmul_time = (end - start) / iterations
        matmul_flops = 2 * (size ** 3)
        matmul_tflops = (matmul_flops / matmul_time) / 1e12
        
        print(f"MatMul (FP32) Average Time: {matmul_time:.6f} s")
        print(f"MatMul (FP32) Performance:  {matmul_tflops:.4f} TFLOPS")
        
        results["metrics"]["matmul_fp32_time_s"] = matmul_time
        results["metrics"]["matmul_fp32_tflops"] = matmul_tflops

        # Benchmark Elementwise
        print(f"Benchmarking Elementwise ({iterations} iters)...")
        start = time.time()
        for _ in range(iterations):
            _ = elementwise(x, y).block_until_ready()
        end = time.time()
        elem_time = (end - start) / iterations
        
        print(f"Elementwise Average Time: {elem_time:.6f} s")
        results["metrics"]["elementwise_time_s"] = elem_time

        # Mixed Precision Benchmark (BFloat16)
        try:
            x_bf16 = x.astype(jnp.bfloat16)
            y_bf16 = y.astype(jnp.bfloat16)
            
            # Warmup BF16
            _ = matmul(x_bf16, y_bf16).block_until_ready()
            
            print(f"Benchmarking MatMul (BFloat16) ({iterations} iters)...")
            start = time.time()
            for _ in range(iterations):
                _ = matmul(x_bf16, y_bf16).block_until_ready()
            end = time.time()
            matmul_bf16_time = (end - start) / iterations
            matmul_bf16_tflops = (matmul_flops / matmul_bf16_time) / 1e12 # Same FLOPS count theoretically
            
            print(f"MatMul (BF16) Average Time: {matmul_bf16_time:.6f} s")
            print(f"MatMul (BF16) Performance:  {matmul_bf16_tflops:.4f} TFLOPS")
            
            results["metrics"]["matmul_bf16_time_s"] = matmul_bf16_time
            results["metrics"]["matmul_bf16_tflops"] = matmul_bf16_tflops
            
        except Exception as e:
            print(f"BFloat16 benchmark skipped: {e}")

        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Benchmark results saved to {save_path}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

def handle_benchmark(args):
    run_benchmark(args.device, args.size, args.iters, args.warmup, args.save)

def handle_config(args):
    if args.show:
        print("Configuration system not initialized yet.")
        # Future: Load and print config from file
    else:
        print("Usage: jaxflow config --show")

def main():
    parser = argparse.ArgumentParser(description="JaxFlow CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    parser.add_argument("-v", "--version", action="version", version=f"JaxFlow {__version__}")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system and library info")
    info_parser.add_argument("--json", action="store_true", help="Output in JSON format")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/gpu/tpu)")
    bench_parser.add_argument("--size", type=int, default=2000, help="Matrix size for benchmark")
    bench_parser.add_argument("--iters", type=int, default=10, help="Number of iterations")
    bench_parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    bench_parser.add_argument("--save", type=str, help="Path to save benchmark results (JSON)")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current config")

    args = parser.parse_args()

    if args.command == "info":
        handle_info(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "config":
        handle_config(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

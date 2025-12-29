# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Replay script for UCMConnector tests.

This script reads a trace file containing time/operation/tokens_num/hash entries
and replays the operations using the same connector setup as uctest_ucm.py.
"""

import json
import os
import time
import asyncio
import secrets
import math
from typing import Dict, List, Tuple, Optional
from unittest.mock import patch

import torch
import pandas as pd
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    KVTransferConfig
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from ucm.integration.vllm.ucm_connector import (
    UCMConnectorMetadata,
    UCMConnector,
    RequestDispatchMeta,
)
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import UcmKVStoreBaseV1


def parse_trace_file(trace_file: str) -> List[Tuple[float, str, int, List[str]]]:
    """
    Parse trace file with JSON format: {"timestamp": ts, "op_type": op, "block_size": size, "blocks": [...]}
    
    Returns:
        List of (timestamp, operation, num_tokens, hashes)
    """
    records = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                timestamp = float(data["timestamp"])
                operation = data["op_type"].strip().lower()
                block_size = int(data["block_size"])
                blocks = data["blocks"]
                num_tokens = len(blocks) * block_size
                records.append((timestamp, operation, num_tokens, blocks))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid line: {line}. Error: {e}")
                continue
    return records


def parse_speed_log(speed_log_file: str) -> List[float]:
    """
    Parse speed log file - each line contains a speed value
    
    Returns:
        List of speed values in order
    """
    speeds = []
    if not os.path.exists(speed_log_file):
        print(f"Warning: Speed log file {speed_log_file} not found")
        return speeds
    
    with open(speed_log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                speed = float(line)
                speeds.append(speed)
            except ValueError:
                continue
    return speeds


def match_speeds_to_operations(
    trace_file: str, 
    speed_log_file: str
) -> Dict[int, float]:
    """
    Match speed values from speed log to operations in trace file
    
    Speed log contains repeated values for each operation (num_req times).
    We extract unique speed values in order and match them to operations.
    
    Returns:
        Dictionary mapping operation index to speed value
    """
    speeds = parse_speed_log(speed_log_file)
    if not speeds:
        return {}
    
    unique_speeds = []
    prev_speed = None
    for speed in speeds:
        if speed != prev_speed:
            unique_speeds.append(speed)
            prev_speed = speed
    
    trace_records = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                trace_records.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    
    operation_speeds = {}
    for op_idx in range(min(len(trace_records), len(unique_speeds))):
        operation_speeds[op_idx] = unique_speeds[op_idx]
    
    return operation_speeds


def make_aligned_tensor(shape, dtype, device, alignment=4096):
    numel = math.prod(shape)
    dtype_size = torch.tensor(1, dtype=dtype).element_size()
    total_bytes = numel * dtype_size

    padded_bytes = total_bytes + alignment
    storage = torch.ByteTensor(padded_bytes).to(device)

    ptr = storage.data_ptr()
    offset = ptr % alignment
    aligned_ptr = ptr + (alignment - offset) if offset != 0 else ptr

    aligned_storage = storage[(aligned_ptr - ptr):].view(dtype)
    tensor = aligned_storage[:numel].view(shape)
    tensor.storage_ref = storage
    return tensor


def make_buffers(
    block_number: int,
    device_id: int,
    batch_size: int,
    head_dim: int,
    block_len: int,
    block_layer: int,
    num_head: int,
    kv: int,
    is_mla: bool,
) -> Tuple[List[bytes], Dict[str, torch.Tensor]]:
    hashes = [secrets.token_bytes(16) for _ in range(block_number)]
    device = f"cuda:{device_id}"
    kv_caches: Dict[str, torch.Tensor] = {}

    for layer in range(block_layer):
        layer_name = f"layer.{layer}"
        if is_mla:
            kv_caches[layer_name] = make_aligned_tensor(
                [block_number, block_len, head_dim],
                dtype=torch.float16,
                device=device,
            )
        else:
            kv_caches[layer_name] = make_aligned_tensor(
                [kv, block_number, block_len, num_head, head_dim],
                dtype=torch.float16,
                device=device,
            )
    return hashes, kv_caches


def build_vllm_config(
    *,
    model_path: str,
    block_size: int,
    num_layers: int,
    num_head: int,
    head_size: int,
    is_mla: bool,
    tp_size: int,
    connector_name: str,
    storage_backends: str,
    transfer_stream_number: int,
    use_direct: bool,
) -> VllmConfig:
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.87,
        swap_space=4,
        cache_dtype="auto",
    )

    # Override HF config to use test parameters instead of model config values
    hf_overrides = {
        "head_dim": head_size,
        "num_key_value_heads": num_head,
        "num_hidden_layers": num_layers,
    }
    if is_mla:
        kv_lora_rank = head_size - 64
        hf_overrides.update({
            "model_type": "deepseek_v3",
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": 64,
        })
    
    model_config = ModelConfig(
        model=model_path,
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        max_model_len=8192,
        max_context_len_to_capture=8192,
        max_logprobs=20,
        disable_sliding_window=False,
        skip_tokenizer_init=True,
        limit_mm_per_prompt={},
        use_async_output_proc=True,
        override_neuron_config={},
        config_format="auto",
        is_deepseek_mla=is_mla,
        hf_overrides=hf_overrides,
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=tp_size,
        worker_use_ray=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_config = DeviceConfig(device=device)

    kv_transfer_config = KVTransferConfig(
        kv_connector="UCMConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connectors": [
                {
                    "ucm_connector_name": connector_name,
                    "ucm_connector_config": {
                        "storage_backends": storage_backends,
                        "use_direct": use_direct,
                        "stream_number": transfer_stream_number,
                        "local_rank_size": 1
                    }
                }
            ]
        },
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        device_config=device_config,
        kv_transfer_config=kv_transfer_config,
    )


def compute_total_bytes(kv_caches: Dict[str, torch.Tensor], batch_size: int, is_mla: bool) -> int:
    total = 0
    for tensor in kv_caches.values():
        if is_mla:
            total += tensor[:batch_size].numel() * tensor.element_size()
        else:
            total += tensor[:, :batch_size].numel() * tensor.element_size()
    return total


def build_forward_context(kv_caches: Dict[str, torch.Tensor], is_mla: bool):
    from dataclasses import dataclass
    from typing import Union

    @dataclass
    class DummyLayer:
        # kv_cache should be indexable (dict or list) for ucm_connector
        kv_cache: Union[Dict[int, torch.Tensor], torch.Tensor, List[torch.Tensor]]

    @dataclass
    class DummyForwardContext:
        no_compile_layers: Dict[str, DummyLayer]
        virtual_engine: int = 0

    layers = {}
    for layer_name, tensor in kv_caches.items():
        if is_mla:
            layers[layer_name] = DummyLayer(kv_cache={0: tensor})
        else:
            layers[layer_name] = DummyLayer(kv_cache={0: tensor})
    return DummyForwardContext(no_compile_layers=layers, virtual_engine=0)


async def process_record(
    connector: UCMConnector,
    scheduler: UcmKVStoreBaseV1,
    hash_map: Dict[str, bytes],
    timestamp: float,
    operation: str,
    num_tokens: int,
    input_hashes: List[str],
    idx: int,
    total: int,
    block_len: int,
    device_id: int,
    head_size: int,
    block_layer: int,
    num_head: int,
    kv: int,
    is_mla: bool,
    original_speed: Optional[float] = None,
) -> Optional[Dict]:
    print(f"\n--- Record {idx+1}/{total}: {operation} at {timestamp} ---")
    
    batch_size = int(num_tokens / block_len)
    real_blocks = batch_size + 10
    
    if operation == "dump":
        generated_hashes, kv_caches = make_buffers(
            real_blocks,
            device_id,
            batch_size,
            head_size,
            block_len,
            block_layer,
            num_head,
            kv,
            is_mla,
        )
        
        # Update hash_map: map log hash (input_hash) to real hash (generated_hashes)
        dump_hashes = generated_hashes[:batch_size]
        for i, input_hash in enumerate(input_hashes):
            if i < len(dump_hashes):
                hash_map[input_hash] = dump_hashes[i]
        
        metadata = UCMConnectorMetadata()
        dump_vllm_block_ids = list(range(batch_size))
        metadata.request_meta["replay_dump"] = RequestDispatchMeta(
            load_block_ids=([], []),
            dump_block_ids=(dump_hashes, dump_vllm_block_ids)
        )
        
        if not hasattr(connector.connector, 'store') or connector.connector.store is None:
            connector.connector.register_kv_caches(kv_caches)
        connector.bind_connector_metadata(metadata)

        total_bytes = compute_total_bytes(kv_caches, batch_size, is_mla)
        start = time.perf_counter()
        connector.wait_for_save()
        write_time = time.perf_counter() - start
        write_bw = (total_bytes / (1024**3)) / write_time if write_time > 0 else 0.0
        
        result = {
            "timestamp": timestamp,
            "operation": "dump",
            "num_tokens": num_tokens,
            "num_blocks": batch_size,
            "time_seconds": write_time,
            "data_size_gb": total_bytes / (1024**3),
            "bandwidth_gbps": write_bw,
        }
        if original_speed is not None:
            result["original_speed"] = original_speed
        
        del kv_caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
        
    elif operation == "load":
        # Get real hashes from mapping
        real_hashes = []
        for input_hash in input_hashes:
            if input_hash in hash_map:
                real_hashes.append(hash_map[input_hash])
            else:
                print(f"Warning: Hash {input_hash} not found in map, skipping")
        
        if not real_hashes:
            print(f"Error: No valid hashes found for load operation")
            return None
        
        _, kv_caches = make_buffers(
            real_blocks,
            device_id,
            batch_size,
            head_size,
            block_len,
            block_layer,
            num_head,
            kv,
            is_mla,
        )

        if not hasattr(connector.connector, 'store') or connector.connector.store is None:
            connector.connector.register_kv_caches(kv_caches)
        
        # Use batch_size hashes for lookup and load
        load_hashes = real_hashes[:batch_size] if len(real_hashes) >= batch_size else real_hashes
        lookup = scheduler.lookup(load_hashes)
        if not all(lookup):
            raise RuntimeError("Found missing cache blocks before load test.")
        
        load_metadata = UCMConnectorMetadata()
        load_vllm_block_ids = list(range(len(load_hashes)))
        load_metadata.request_meta["replay_load"] = RequestDispatchMeta(
            load_block_ids=(load_hashes, load_vllm_block_ids),
            dump_block_ids=([], [])
        )
        connector.bind_connector_metadata(load_metadata)

        forward_context = build_forward_context(kv_caches, is_mla)

        total_bytes = compute_total_bytes(kv_caches, len(load_hashes), is_mla)
        start = time.perf_counter()
        connector.start_load_kv(forward_context)
        read_time = time.perf_counter() - start
        read_bw = (total_bytes / (1024**3)) / read_time if read_time > 0 else 0.0
        
        result = {
            "timestamp": timestamp,
            "operation": "load",
            "num_tokens": num_tokens,
            "num_blocks": len(load_hashes),
            "time_seconds": read_time,
            "data_size_gb": total_bytes / (1024**3),
            "bandwidth_gbps": read_bw,
        }
        if original_speed is not None:
            result["original_speed"] = original_speed
        
        del kv_caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
        
    elif operation == "lookup":
        real_hashes = []
        for input_hash in input_hashes:
            if input_hash in hash_map:
                real_hashes.append(hash_map[input_hash])
            else:
                print(f"Warning: Hash {input_hash} not found in map, converting from hex")
                # Convert hex string to bytes (assuming input_hash is hex encoded)
                try:
                    real_hashes.append(bytes.fromhex(input_hash))
                except ValueError:
                    print(f"Error: Cannot convert hash {input_hash} from hex, skipping")
                    continue
        
        if not real_hashes:
            print(f"Error: No valid hashes found for lookup operation")
            return None
        
        # Perform lookup operation using scheduler
        start_time = time.perf_counter()
        founds = scheduler.lookup(real_hashes)
        elapsed_time = time.perf_counter() - start_time
        
        hit_count = sum(1 for f in founds if f)
        miss_count = len(founds) - hit_count
        
        print(
            f"LOOKUP: {len(real_hashes)} blocks, "
            f"Hit: {hit_count}, Miss: {miss_count}, "
            f"Time={elapsed_time:.4f} s"
        )
        
        result = {
            "timestamp": timestamp,
            "operation": "lookup",
            "num_tokens": num_tokens,
            "num_blocks": len(real_hashes),
            "time_seconds": elapsed_time,
            "hit_count": hit_count,
            "miss_count": miss_count,
            "hit_rate": hit_count / len(real_hashes) if real_hashes else 0.0,
        }
        if original_speed is not None:
            result["original_speed"] = original_speed
        return result
    else:
        print(f"Unknown operation: {operation}, skipping")
        return None


async def schedule_task(
    connector: UCMConnector,
    scheduler: UcmKVStoreBaseV1,
    hash_map: Dict[str, bytes],
    relative_time: float,
    operation: str,
    num_tokens: int,
    input_hashes: List[str],
    original_timestamp: float,
    idx: int,
    total: int,
    start_time: float,
    block_len: int,
    device_id: int,
    head_size: int,
    block_layer: int,
    num_head: int,
    kv: int,
    is_mla: bool,
    original_speed: Optional[float] = None,
) -> Optional[Dict]:
    current_time = time.perf_counter() - start_time
    wait_time = relative_time - current_time
    
    if wait_time > 0:
        print(f"Record {idx+1}: Waiting {wait_time:.4f} seconds (relative time: {relative_time:.4f}s)...")
        await asyncio.sleep(wait_time)
    else:
        print(f"Record {idx+1}: Executing immediately (relative time {relative_time:.4f}s already passed)")
    
    return await process_record(
        connector=connector,
        scheduler=scheduler,
        hash_map=hash_map,
        timestamp=original_timestamp,
        operation=operation,
        num_tokens=num_tokens,
        input_hashes=input_hashes,
        idx=idx,
        total=total,
        block_len=block_len,
        device_id=device_id,
        head_size=head_size,
        block_layer=block_layer,
        num_head=num_head,
        kv=kv,
        is_mla=is_mla,
        original_speed=original_speed,
    )


async def main_async():
    trace_file = "temp/ucm_ops.log"  # Trace file with time/operation/tokens_num/hash format
    speed_log_file = os.path.join(os.path.dirname(__file__), "temp/ucm_ops_speed.log")
    output = "ucm_ops_2.xlsx"
    storage_backends = "/home/zht/zht_3/test_data/ucm_data"
    device_id = 0 if torch.cuda.is_available() else -1
    transfer_stream_number = 32
    model_path = "/home/models/QwQ-32B"
    
    is_mla = False  
    use_direct = False
    
    if is_mla:
        block_layer = 27
        head_size = 576
        kv = 1
        block_len = 64
        num_head = 1
    else:
        block_layer = 64
        head_size = 128
        kv = 2
        block_len = 128
        num_head = 4
    
    block_dim = head_size * num_head
    io_size = block_dim * block_len * 2  # 2 bytes per float16 element
    block_size = io_size * block_layer
    total_tp_size = 1

    # Build vLLM configuration
    vllm_config = build_vllm_config(
        model_path=model_path,
        block_size=block_len,
        num_layers=block_layer,
        num_head=num_head,
        head_size=head_size,
        is_mla=is_mla,
        tp_size=total_tp_size,
        connector_name="UcmNfsStore",
        storage_backends=storage_backends,
        transfer_stream_number=transfer_stream_number,
        use_direct=use_direct,
    )

    storage_backends_list = [os.path.join(path, "kv") for path in storage_backends.split(":") if path]
    
    ucm_connector_name = "UcmNfsStore"  # Use the same connector name as in vllm_config
    scheduler_config = {
        "storage_backends": storage_backends_list,
        "block_size": block_size,
        "device_id": -1,  # device_id=-1 means transferEnable=false
        "tensor_size": io_size,
        "stream_number": transfer_stream_number,
        "io_direct": use_direct,
        "unique_id": secrets.token_hex(8),
    }
    scheduler = UcmConnectorFactoryV1.create_connector(ucm_connector_name, scheduler_config)

    # Create connector
    dummy_world_group = type("DummyWorldGroup", (), {"local_rank": 0})()
    
    # Create a dummy TP group to avoid initialization error
    class DummyTPGroup:
        def broadcast(self, tensor, src):
            pass
            
        @property
        def rank(self):
            return 0

    dummy_tp_group = DummyTPGroup()
    
    with patch(
        "ucm.integration.vllm.ucm_connector.get_world_group",
        return_value=dummy_world_group,
    ), patch(
        "ucm.integration.vllm.ucm_connector.get_tp_group",
        return_value=dummy_tp_group,
    ):
        connector = UCMConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector.rank = device_id if device_id >= 0 else 0
    connector.connector.kv_caches = {}

    # Parse trace file
    records = parse_trace_file(trace_file)
    print(f"Loaded {len(records)} records from {trace_file}")

    operation_speeds = match_speeds_to_operations(trace_file, speed_log_file)
    print(f"Loaded {len(operation_speeds)} speed values from {speed_log_file}")

    hash_map: Dict[str, bytes] = {}
    
    results = []
    
    if not records:
        print("No records to process")
        return
    
    # Convert timestamps to relative times
    base_timestamp = records[0][0]
    print(f"Base timestamp: {base_timestamp}, using relative time intervals")
    
    relative_records = []
    for idx, (timestamp, operation, num_tokens, input_hashes) in enumerate(records):
        relative_time = timestamp - base_timestamp
        original_speed = operation_speeds.get(idx)
        relative_records.append((relative_time, operation, num_tokens, input_hashes, timestamp, original_speed))
        print(f"Record {idx+1}: absolute_time={timestamp}, relative_time={relative_time:.4f}s, original_speed={original_speed}")
    
    start_time = time.perf_counter()
    
    # Schedule and execute tasks
    tasks = []
    for idx, (relative_time, operation, num_tokens, input_hashes, original_timestamp, original_speed) in enumerate(relative_records):
        task = schedule_task(
            connector=connector,
            scheduler=scheduler,
            hash_map=hash_map,
            relative_time=relative_time,
            operation=operation,
            num_tokens=num_tokens,
            input_hashes=input_hashes,
            original_timestamp=original_timestamp,
            idx=idx,
            total=len(relative_records),
            start_time=start_time,
            block_len=block_len,
            device_id=device_id,
            head_size=head_size,
            block_layer=block_layer,
            num_head=num_head,
            kv=kv,
            is_mla=is_mla,
            original_speed=original_speed,
        )
        tasks.append(task)
    
    # Execute all tasks concurrently
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in task_results:
        if isinstance(result, Exception):
            print(f"Task failed: {result}")
        elif result is not None:
            results.append(result)
    
    results.sort(key=lambda x: x["timestamp"])

    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(os.path.dirname(__file__), output)
        if os.path.exists(output_path):
            existing_df = pd.read_excel(output_path, engine="openpyxl")
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"\nResults saved to {output_path}")
        print(f"\nSummary:")
        print(f"  Total records: {len(results)}")
        print(f"  Dump operations: {len([r for r in results if r['operation'] == 'dump'])}")
        print(f"  Load operations: {len([r for r in results if r['operation'] == 'load'])}")
        print(f"  Lookup operations: {len([r for r in results if r['operation'] == 'lookup'])}")
    else:
        print("No results to save")

    del connector


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
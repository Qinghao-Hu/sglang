"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.eagle_utils import EagleDraftInput
import csv


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    # batch_size: Tuple[int] = (1,)
    # input_len: Tuple[int] = (1024,)
    # output_len: Tuple[int] = (16,)
    batch_size: int = 1
    input_len: int = 1024
    output_len: int = 16
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    profile: bool = False
    profile_filename_prefix: str = "profile"
    result_csv_filename: str = "spec_benchmark_results.csv"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--profile", action="store_true", help="Use Torch Profiler."
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument("--result-csv-filename", type=str, default=BenchArgs.result_csv_filename)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def write_results_to_csv(csv_filename, results, tp_size, spec_num_steps):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as csvfile:
        fieldnames = [
            'run_name', 'cuda_graph', 'batch_size', 'tp_size', 'input_len', 'output_len',
            'sim_acc_tokens', 'exact_avg_acc_tokens', 'spec_num_steps', 'tokens_to_verify', 
            'prefill_latency', 'avg_decode_latency',
            'prefill_throughput', 'avg_decode_throughput',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for res in results:
            writer.writerow({
                'run_name': res['run_name'],
                'cuda_graph': os.environ.get("ALLOW_CUDA_GRAPH"),
                'batch_size': res['batch_size'],
                'tp_size': tp_size,
                'input_len': res['input_len'],
                'output_len': res['output_len'],
                'sim_acc_tokens': res.get('sim_acc_tokens', ''),
                'exact_avg_acc_tokens': res.get('exact_avg_acc_tokens', ''),
                'spec_num_steps': spec_num_steps,
                'tokens_to_verify': res.get('tokens_to_verify', ''),
                'prefill_latency': res['prefill_latency'],
                'avg_decode_latency': res.get('avg_decode_latency', ''),
                'prefill_throughput': res['prefill_throughput'],
                'avg_decode_throughput': res.get('avg_decode_throughput', ''),
            })


def load_model(server_args, port_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # NOTE (Shang): Init Eagle Draft Model here.
    # NOTE (Shang): dp is not considered yet
    tp_worker = TpModelWorker(
        server_args=server_args,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        dp_rank=0,
        nccl_port=port_args.nccl_port,
    )

    draft_worker = EAGLEWorker(
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        server_args=server_args,
        nccl_port=port_args.nccl_port,
        target_worker=tp_worker,
        dp_rank=0,
    )

    rank_print(f"max_total_num_tokens={tp_worker.model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return tp_worker.model_runner, tokenizer, draft_worker


def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=bench_args.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len, bench_args):
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=bench_args.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        reqs.append(req)

    return reqs


@torch.no_grad
def extend(reqs, model_runner, draft_worker, server_args): # Prefill
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.from_string(server_args.speculative_algorithm),
        enable_custom_logit_processor=False,
    )

    batch.prepare_for_extend()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    (
        logits_output,
        next_token_ids,
        bid,
        num_accepted_tokens,
    ) = draft_worker.forward_batch_speculative_generation(batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits

@torch.no_grad
def sim_verify(batch_size, input_token_ids, batch, model_runner, draft_worker, server_args):
    batch.output_ids = input_token_ids
    batch.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    batch.prepare_for_decode()
    (
        logits_output,
        next_token_ids,
        bid,
        num_accepted_tokens,
    ) = draft_worker.forward_batch_speculative_generation(batch)
  
    batch.output_ids = next_token_ids
    
    new_gen_tokens = (num_accepted_tokens // batch_size) + 1

    return next_token_ids, logits_output.next_token_logits, new_gen_tokens

def correctness_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer, draft_worker = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner, draft_worker)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner, draft_worker)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    torch.get_device_module(device).synchronize()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    profile,
    profile_filename_prefix,
    draft_worker,
    tokens_to_verify,
    server_args,
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "tokens_to_verify": tokens_to_verify,
    }

    tot_latency = 0

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()

    # Prefill
    synchronize(device)
    tic = time.time()
    next_token_ids, _, batch = extend(reqs, model_runner, draft_worker, server_args)
    synchronize(device)
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode
    decode_latencies = []
    decode_acc_lens = []
    token_idx = 0
    iter_round = 0
    while token_idx < output_len:
        synchronize(device)
        tic = time.time()

        next_token_ids, _, new_gen_tokens = sim_verify(batch_size, next_token_ids, batch, model_runner, draft_worker, server_args)

        token_idx += new_gen_tokens

        synchronize(device)
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size * new_gen_tokens / latency  
        decode_latencies.append(latency)
        decode_acc_lens.append(new_gen_tokens)
        if iter_round < 5:
            rank_print(
                f"Decode.  latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )
        iter_round +=1

    if profile:
        profiler.stop()
        profile_filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}.trace.json.gz"
        parent_dir = os.path.dirname(os.path.abspath(profile_filename))
        os.makedirs(parent_dir, exist_ok=True)
        profiler.export_chrome_trace(profile_filename)
        rank_print(f"torch profiler chrome trace saved to {profile_filename}")

    # Record decode timing from 2nd output
    # if output_len > 1:
    #     med_decode_latency = np.median(decode_latencies)
    #     med_decode_throughput = batch_size * max_new_gen_tokens / med_decode_latency    # use max_new_gen_tokens for throughput calculation avoid the effect of last output token length's effect
    #     rank_print(
    #         f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
    #     )
    #     measurement_results["median_decode_latency"] = med_decode_latency
    #     measurement_results["median_decode_throughput"] = med_decode_throughput
    #     measurement_results["new_gen_tokens"] = max_new_gen_tokens

    if output_len > 1:  # Average decode latency stats
        tot_decode_latency = tot_latency - prefill_latency
        avg_decode_latency = np.mean(decode_latencies)
        assert np.sum(decode_acc_lens) == output_len, f"Sum of decoded tokens should be equal to the output_len, now {np.sum(decode_acc_lens)} != {output_len}"
        avg_decode_throughput = batch_size * output_len / tot_decode_latency
        rank_print(
            f"Decode.  average latency (per_step): {avg_decode_latency:6.5f} s, average throughput: {avg_decode_throughput:9.2f} token/s"
        )
        measurement_results["avg_decode_latency"] = avg_decode_latency
        measurement_results["avg_decode_throughput"] = avg_decode_throughput
        measurement_results["sim_acc_tokens"] = float(os.environ.get("SIMULATE_ACC_LEN"))
        measurement_results["exact_avg_acc_tokens"] = np.mean(decode_acc_lens)


    throughput = (input_len + output_len) * batch_size / tot_latency
    # rank_print(
    #     f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    # )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer, draft_worker = load_model(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size, bench_args.input_len, bench_args
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size,
        bench_args.input_len,
        # 8,  # shorter decoding to speed up the warmup
        bench_args.output_len,
        server_args.device,
        profile=False,
        profile_filename_prefix="",  # not used
        draft_worker=draft_worker,
        tokens_to_verify=server_args.speculative_num_draft_tokens,
        server_args=server_args,
    )

    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    # for bs, il, ol in itertools.product(
    #     bench_args.batch_size, bench_args.input_len, bench_args.output_len
    # ):
    bs, il, ol = (bench_args.batch_size, bench_args.input_len, bench_args.output_len)
    reqs = prepare_synthetic_inputs_for_latency_test(bs, il, bench_args)
    ret = latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bs,
        il,
        ol,
        server_args.device,
        bench_args.profile if tp_rank == 0 else None,
        bench_args.profile_filename_prefix,
        draft_worker,
        server_args.speculative_num_draft_tokens,
        server_args,
    )
    if ret is not None:
        result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")
    
    if tp_rank == 0 and bench_args.result_csv_filename:
        write_results_to_csv(bench_args.result_csv_filename, result_list, server_args.tp_size, server_args.speculative_num_steps)


def main(server_args, bench_args):
    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)

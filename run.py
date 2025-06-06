import yaml
import os
import json
import logging
import argparse
import re
import socket, threading
from time import sleep
from multiprocessing.pool import ThreadPool
import torch
import requests
import time

SLEEP_TIME = 20
TIMEOUT = 100

class ThreadWithReturnValue(threading.Thread):
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        super().join()
        return self._return

def is_server_ready(base_url="http://127.0.0.1:8000/v1/models"):
    start = time.time()
    while True:
        try:
            response = requests.get(base_url)
            if response.status_code == 200:
                print(f"{base_url} ready")
                return True
            else:
                print(f"{base_url} not ready, wait for {SLEEP_TIME} seconds...")
                time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"{base_url} not ready with error {e}, wait for {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
        if time.time() - start > TIMEOUT:
            print(f"[WARN] timed out for waiting vLLM API server ready")
            break
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="config file path for evaluation")
    
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    OutputHWUsage = config['OutputHWUsage']

    required_ttft = config['PerfRequirement']['TTFT']
    required_tpot = config['PerfRequirement']['TPOT']

    save_dir = config["Logging"]["save_dir"]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=save_dir, format="%(asctime)s %(message)s", level=logging.INFO)
    logger.info("Save logging info to %s", save_dir)
    print("Save logging info to %s", save_dir)
    
    logger.info("config file: %s", config)
    # device
    device = config["Device"]
    if device != "gcu":
        assert "Only Support Evaluation on GCU !!!"

    # Model
    model_name = config["Model"]["model_name"]
    model_path = config["Model"]["model_path"]
    tokenizer_path = config["Model"]["tokenizer_path"]

    # Dataset:
    if not config["Dataset"]["use_random"]:
        data_name = config["Dataset"]["data_name"]
        data_path = config["Dataset"]["data_path"]
    else:
        use_random_data = True

    # DeployParam
    DeployParam = config["DeployParam"]
    gpu_memory_utilization = DeployParam["gpu_memory_utilization"]
    tensor_parallel_size = DeployParam["tensor_parallel_size"]
    pipeline_parallel_size = DeployParam["pipeline_parallel_size"]
    batch_size = num_prompts = DeployParam["batch_size"]
    trust_remote_code = DeployParam["trust_remote_code"]
    block_size = DeployParam["block_size"]
    enable_chunked_prefill = DeployParam["enable_chunked_prefill"]
    enable_prefix_caching = DeployParam["enable_prefix_caching"]
    disable_async_output_proc = DeployParam["disable_async_output_proc"]
    enforce_eager = DeployParam["enforce_eager"]
    distributed_executor_backend = DeployParam["distributed_executor_backend"]
    max_num_seqs = DeployParam["max_num_seqs"]

    # ModelKwargs
    ModelKwargs = config["ModelKwargs"]
    dtype = ModelKwargs["dtype"]
    max_model_len = ModelKwargs["max_model_len"]
    max_seq_len = ModelKwargs["max_model_len"]
    quantization = ModelKwargs["quantization"]

    # SamplingParams
    SamplingParams = config["SamplingParams"]
    temperature = SamplingParams["temperature"]
    max_tokens = max_out_len = max_gen_tokens = SamplingParams["max_tokens"]
    output_len = SamplingParams["output_len"]
    top_p = SamplingParams["top_p"]
    top_k = SamplingParams["top_k"]
    ignore_eos = SamplingParams["ignore_eos"]
    num_seqs = SamplingParams["num_seqs"]
    keep_special_tokens = SamplingParams["keep_special_tokens"]
    strict_in_out_len = SamplingParams["strict_in_out_len"]

    MetricType = config["MetricType"]
    EvalTool = config["EvalTool"]
    InferType = config["InferType"]

    logging.info("MetricType: %s", MetricType)
    logging.info("EvalTool: %s", EvalTool)
    logging.info("InferType: %s", InferType)
    print("MetricType: %s", MetricType)
    print("EvalTool: %s", EvalTool)
    print("InferType: %s", InferType)

    if MetricType == "Acc":
        if EvalTool == "BenchmarkTest":
            cmd = f"python3 -m vllm_utils.benchmark_test --acc \
                    --disable-log-stats \
                    --model={model_path}  \
                    --device={device} \
                    --tokenizer={tokenizer_path} \
                    --gpu-memory-utilization={gpu_memory_utilization} \
                    --dtype={dtype} \
                    --max-model-len={max_model_len} \
                    --dataset={data_path} \
                    --temperature={temperature} \
                    --tensor_parallel_size={tensor_parallel_size} \
                    --n={num_seqs} \
                    --output-len={output_len} " + \
                    "--trust_remote_code " * trust_remote_code + \
                    "--enable_chunked_prefill " * enable_chunked_prefill + \
                    "--enable_prefix_caching " * enable_prefix_caching + \
                    "--disable_async_output_proc " * disable_async_output_proc + \
                    "--enforce_eager" * enforce_eager
            os.system(cmd)
            logging.info("Run command: %s", cmd)
            print("Run command: %s", cmd)
        elif EvalTool == "lm_eval":
            lm_model = config["Acc"]["lm_eval"]["model"]
            tasks = config["Acc"]["lm_eval"]["tasks"]
            if InferType == "offline":
                num_fewshot = config["Acc"]["lm_eval"]["num_fewshot"]
                cmd = f"lm_eval --model {lm_model} \
                        --model_args pretrained={model_path},tensor_parallel_size={tensor_parallel_size},dtype={dtype},gpu_memory_utilization={gpu_memory_utilization},max_model_len={max_model_len},trust_remote_code={trust_remote_code}" + ",quantization={quantization} "*(quantization!=None) + \
                        f" --tasks {tasks} \
                        --batch_size {batch_size} \
                        --num_fewshot {num_fewshot} "+ \
                        "--output_path {save_dir} \
                        --log_samples \
                        --seed 0 \
                        --verbosity DEBUG \
                        --show_config \
                        --gen_kwargs temperature={temperature},top_k={top_k},top_p={top_p},max_gen_toks={max_gen_tokens}"
            if InferType == "serving":
                # launch serving
                serving_cmd = f"python3 -m vllm.entrypoints.openai.api_server \
                              --model {model_path} \
                              --dtype {dtype} \
                              --device={device} \
                              --block-size {block_size} " + \
                              "--trust_remote_code " * trust_remote_code + \
                              f"--max-model-len {max_model_len} \
                              --pipeline-parallel-size {pipeline_parallel_size} \
                              --tensor-parallel-size {tensor_parallel_size} \
                              --pipeline-parallel-size {pipeline_parallel_size} " + \
                              f"--distributed-executor-backend {distributed_executor_backend}" * (distributed_executor_backend!=None) + \
                              "--enable-prefix-caching " * enable_prefix_caching + \
                              f"--gpu-memory-utilization {gpu_memory_utilization} \
                              --max-num-seqs {max_num_seqs} \
                              --quantization {quantization} " + \
                              "--enforce-eager" * enforce_eager + \
                              "--enable-chunked-prefill " * enable_chunked_prefill
                logging.info("Run serving command: %s", serving_cmd)
                print("Run serving command: %s", serving_cmd)
                def server_func():
                    os.system(serving_cmd)
                t1 = threading.Thread(target=server_func)
                t1.start()

                if not is_server_ready():
                    print("serving failed...")
                    logging.info("serving failed...")
                    return

                # launch lm_eval
                base_url = config["Acc"]["lm_eval"]["base_url"]
                apply_chat_template = config["Acc"]["lm_eval"]["apply_chat_template"]
                num_concurrent = config["Acc"]["lm_eval"]["num_concurrent"]
                cmd = f"lm_eval --model {lm_model} \
                        --model_args model={model_path},base_url={base_url},num_concurrent={num_concurrent},max_retries=5,max_length={max_model_len},trust_remote_code={trust_remote_code} \
                        --tasks {tasks} \
                        --batch_size {batch_size} " + \
                        "--trust_remote_code " * trust_remote_code + \
                        "--apply_chat_template " * apply_chat_template + \
                        "--output_path {save_dir} \
                        --seed 0 \
                        --verbosity DEBUG \
                        --show_config \
                        --gen_kwargs temperature={temperature},top_k={top_k},top_p={top_p},max_gen_toks={max_gen_tokens}"
                logging.info("Run command: %s", cmd)
                print("Run command: %s", cmd)
                def func():
                    os.system(cmd)
                t2 = threading.Thread(target=func)
                t2.start()
                logging.info("Run command: %s", cmd)
                print("Run command: %s", cmd)
        elif EvalTool == "OpenCompass":
            datasets = config["Acc"]["OpenCompass"]["datasets"]
            cmd = f"python3 -m vllm_utils.evaluate_datasets.run \
                    --datasets {datasets} \
                    --data-dir {data_path} \
                    --vllm-path {model_path} \
                    --work-dir ./ \
                    --tensor-parallel-size {tensor_parallel_size} \
                    --batch-size {batch_size} \
                    --max-seq-len {max_seq_len} \
                    --max-out-len {max_out_len} \
                    --device {device} \
                    --model-kwargs dtype={dtype} max_model_len={max_model_len} block_size={block_size} gpu_memory_utilization={gpu_memory_utilization} \
                    --generation-kwargs temperature={temperature},top_k={top_k},top_p={top_p},max_gen_toks={max_gen_tokens}"
            os.system(cmd)
            logging.info("Run command: %s", cmd)
            print("Run command: %s", cmd)

    elif MetricType == "Perf":
        if EvalTool == "BenchmarkTest":
            input_len =  config["Perf"]["BenchmarkTest"]["input_len"]
            cmd = f"python3 -m vllm_utils.benchmark_test --perf \
                    --model {model_path} \
                    --tensor-parallel-size {tensor_parallel_size} \
                    --max-model-len {max_model_len} \
                    --input-len {input_len} \
                    --output-len {output_len} \
                    --dtype={dtype} \
                    --device {device} \
                    --num-prompts {num_prompts} \
                    --block-size={block_size} " + \
                    "--trust-remote_code " * trust_remote_code + \
                    "--enable-chunked-prefill " * enable_chunked_prefill + \
                    "--enable-prefix-caching " * enable_prefix_caching + \
                    "--disable-async-output-proc " * disable_async_output_proc + \
                    "--enforce-eager" * enforce_eager
            logging.info("Run command: %s", cmd)
            print("Run command: %s", cmd)
            os.system(cmd)
        if EvalTool == "BenchmarkServing":
            input_len =  config["Perf"]["BenchmarkServing"]["input_len"]
            dataset_name = config["Perf"]["BenchmarkServing"]["dataset_name"]
            dataset_path = config["Perf"]["BenchmarkServing"]["dataset_path"]
            request_rate = config["Perf"]["BenchmarkServing"]["request_rate"]
            max_concurrency = config["Perf"]["BenchmarkServing"]["max_concurrency"]
            host = config["Perf"]["BenchmarkServing"]["host"]
            port = config["Perf"]["BenchmarkServing"]["port"]
            endpoint = config["Perf"]["BenchmarkServing"]["endpoint"]
            serving_cmd = f"python3 -m vllm.entrypoints.openai.api_server \
                           --model {model_path} \
                           --dtype {dtype} \
                           --device={device} \
                           --block-size {block_size} " + \
                           "--trust_remote_code " * trust_remote_code + \
                           f"--max-model-len {max_model_len} \
                           --pipeline-parallel-size {pipeline_parallel_size} \
                           --tensor-parallel-size {tensor_parallel_size} \
                           --pipeline-parallel-size {pipeline_parallel_size} " + \
                           f"--distributed-executor-backend {distributed_executor_backend}" * (distributed_executor_backend!=None) + \
                           "--enable-prefix-caching " * enable_prefix_caching + \
                           f"--gpu-memory-utilization {gpu_memory_utilization} \
                           --max-num-seqs {max_num_seqs} \
                           --quantization {quantization} " + \
                           "--enforce-eager" * enforce_eager + \
                           "--enable-chunked-prefill " * enable_chunked_prefill
            logging.info("Run serving command: %s", serving_cmd)
            print("Run serving command: %s", serving_cmd)
            def server_func():
                os.system(serving_cmd)
            t1 = threading.Thread(target=server_func)
            t1.start()
            #sleep(SLEEP_TIME)

            if not is_server_ready():
                print("serving failed...")
                logging.info("serving failed...")
                return

            cmd = f"python3 -m vllm_utils.benchmark_serving \
                    --backend vllm \
                    --model {model_path} \
                    --dataset-name {dataset_name} " + \
                    "--dataset-path {dataset_path}" * (dataset_path!=None) + \
                    f"--request-rate {request_rate} \
                    --num-prompts {num_prompts} \
                    --host {host} \
                    --port {port} \
                    --endpoint {endpoint} " + \
                    f"--max-concurrency {max_concurrency}" * (max_concurrency!=None) + \
                    "--{dataset_name}-input-len {input_len} \
                    --{dataset_name}-output-len {output_len} " + \
                    "--ignore_eos " * ignore_eos \
                    + "--strict-in-out-len " * strict_in_out_len + \
                    "--keep_special_tokens " * keep_special_tokens + \
                    "--trust-remote_code " * trust_remote_code
            logging.info("run command: %s", cmd)
            print("run command: %s", cmd)
            
            def func():
                os.system(cmd)
            t2 = threading.Thread(target=func)
            t2.start()
            logging.info("Run command: %s", cmd)
            print("Run command: %s", cmd)
    
        if EvalTool == "EvalScope":
            url = config["Perf"]["EvalScope"]["url"]
            parallel = config["Perf"]["EvalScope"]["parallel"]
            number = config["Perf"]["EvalScope"]["number"]
            api = config["Perf"]["EvalScope"]["api"]
            dataset = config["Perf"]["EvalScope"]["dataset"]
            stream = config["Perf"]["EvalScope"]["stream"]
            max_prompt_length = config["Perf"]["EvalScope"]["max_prompt_length"]
            min_prompt_length = config["Perf"]["EvalScope"]["min_prompt_length"]
            # launch serving
            serving_cmd = f"python3 -m vllm.entrypoints.openai.api_server \
                          --model {model_path} \
                          --dtype {dtype} \
                          --device={device} \
                          --block-size {block_size} " + \
                          "--trust_remote_code " * trust_remote_code + \
                          f"--max-model-len {max_model_len} \
                          --pipeline-parallel-size {pipeline_parallel_size} \
                          --tensor-parallel-size {tensor_parallel_size} \
                          --pipeline-parallel-size {pipeline_parallel_size} " + \
                          f"--distributed-executor-backend {distributed_executor_backend}" * (distributed_executor_backend!=None) + \
                          "--enable-prefix-caching " * enable_prefix_caching + \
                          f"--gpu-memory-utilization {gpu_memory_utilization} \
                          --max-num-seqs {max_num_seqs} \
                          --quantization {quantization} " + \
                          "--enforce-eager" * enforce_eager + \
                          "--enable-chunked-prefill " * enable_chunked_prefill
            logging.info("Run serving command: %s", serving_cmd)
            print("Run serving command: %s", serving_cmd)
            def server_func():
                os.system(serving_cmd)
            t1 = threading.Thread(target=server_func)
            t1.start()
            #sleep(SLEEP_TIME)
            
            if not is_server_ready():
                print("serving failed...")
                logging.info("serving failed...")
                return
            
            cmd = f"evalscope perf \
                    --tokenizer-path {tokenizer_path} \
                    --max-prompt-length {max_prompt_length} \
                    --min-prompt-length {min_prompt_length}  \
                    --url {url} \
                    --parallel {parallel} \
                    --model {model_path} \
                    --number {number} \
                    --api {api} \
                    --dataset {dataset} \
                    --max-tokens {max_tokens} \
                    --top-p {top_p} \
                    --top-k {top_k} " + \
                    "--stream"*stream
            logging.info("run command: %s", cmd)
            print("run command: %s", cmd)
            def func():
                os.system(cmd)
            t2 = threading.Thread(target=func)
            t2.start()
            logging.info("Run command: %s", cmd)
            print("Run command: %s", cmd)

    elif MetricType == "PerfTunning":
        # Fixex params: input_len, output_len, etc.
        # Params for tunning: num_prompts, enable_chunked_prefill, enable_prefix_caching, disable_async_output_proc, enforce_eager
        if EvalTool == "BenchmarkTest":
            input_len =  config["Perf"]["BenchmarkTest"]["input_len"]
            max_num_prompts = config["PerfTunning"]["BenchmarkTest"]["max_num_prompts"]
            min_num_prompts = config["PerfTunning"]["BenchmarkTest"]["min_num_prompts"]
            grid_num_prompts = config["PerfTunning"]["BenchmarkTest"]["grid_num_prompts"]
            results = {}
            results["perf_results"] = {}
            
            # find the best num_prompts
            best_throughput = -1
            best_num_prompts = -1
            for _num_prompts in range(min_num_prompts, max_num_prompts+grid_num_prompts, grid_num_prompts):
                cmd = f"python3 -m vllm_utils.benchmark_test --perf \
                        --model {model_path}  \
                        --tensor-parallel-size {tensor_parallel_size} \
                        --max-model-len {max_model_len} \
                        --input-len {input_len} \
                        --output-len {output_len} \
                        --dtype={dtype} \
                        --device {device} \
                        --num-prompts {_num_prompts} \
                        --block-size={block_size} " + \
                        "--trust_remote_code " * trust_remote_code
                output_str = os.popen(cmd).read()
                start = [i.start() for i in re.finditer('{', output_str)][-1]
                end = [i.start() for i in re.finditer('}', output_str)][-1]
                perf_result = json.loads(output_str[start:(end+1)])
                
                if float(perf_result["prefill_latency_per_token"].split(" ")[0]) <= required_ttft and \
                        float(perf_result["latency_per_token"].split(" ")[0]) <= required_tpot:
                    results["perf_results"][_num_prompts] = perf_result
                    if float(results["perf_results"][_num_prompts]["Throughput"][1].split(" ")[0]) >= best_throughput:
                        best_num_prompts = _num_prompts
                        best_throughput = float(results["perf_results"][_num_prompts]["Throughput"][1].split(" ")[0])
                        logging.info("Find better num_prompts: " + str(best_num_prompts) + ", throughput: " + str(best_throughput))
            results["best_num_prompts"] = best_num_prompts
            logging.info("best_num_prompts: %s", str(best_num_prompts))
            print("best_num_prompts: %s", str(best_num_prompts))

            # find the best combination of the following EngineArgs
            all_choices = [("--enable-chunked-prefill ", enable_chunked_prefill), ("--enable-prefix-caching ",enable_prefix_caching), ("--disable-async-output-proc ",disable_async_output_proc), ("--enforce-eager", enforce_eager)]
            choices = []
            for i in all_choices:
                if i[1]:
                    choices.append(i[0])
            cmd = f"python3 -m vllm_utils.benchmark_test --perf \
                    --model {model_path}  \
                    --tensor-parallel-size {tensor_parallel_size} \
                    --max-model-len {max_model_len} \
                    --input-len {input_len} \
                    --output-len {output_len} \
                    --dtype={dtype} \
                    --device {device} \
                    --num-prompts {best_num_prompts} \
                    --block-size={block_size} " + \
                    "--trust_remote_code " * trust_remote_code
            selected_choices = []
            for choice in choices:
                cmd += choice
                output_str = os.popen(cmd).read()
                start = [i.start() for i in re.finditer('{', output_str)][-1]
                end = [i.start() for i in re.finditer('}', output_str)][-1]
                perf_result = json.loads(output_str[start:(end+1)])
                if float(perf_result["prefill_latency_per_token"].split(" ")[0]) >= required_ttft and \
                        float(perf_result["latency_per_token"].split(" ")[0]) >= required_tpot:
                    cmd.replace(choice, "")
                    continue
                if float(perf_result["Throughput"][1].split(" ")[0]) >= best_throughput:
                    selected_choices.append(choice)
                    logging.info("find better choices: %s", str(selected_choices))
                    print("find better choices: %s", str(selected_choices))
                    best_throughput = float(perf_result["Throughput"][1].split(" ")[0])
                    results["perf_results"][_num_prompts] = perf_result
                    results["best_cmd"] = cmd
                else:
                    cmd.replace(choice, "")
            logging.info("**Final result:** \n")
            logging.info(str(results))
            print("**Final result:** \n")
            print(str(results))
        
        if EvalTool == "BenchmarkServing":
            input_len =  config["Perf"]["BenchmarkServing"]["input_len"]
            dataset_name = config["Perf"]["BenchmarkServing"]["dataset_name"]
            dataset_path = config["Perf"]["BenchmarkServing"]["dataset_path"]
            request_rate = config["Perf"]["BenchmarkServing"]["request_rate"]
            max_concurrency = config["Perf"]["BenchmarkServing"]["max_concurrency"]
            host = config["Perf"]["BenchmarkServing"]["host"]
            port = config["Perf"]["BenchmarkServing"]["port"]
            endpoint = config["Perf"]["BenchmarkServing"]["endpoint"]
            min_request_rate = config["PerfTunning"]["BenchMarkServing"]["min_request_rate"]
            max_request_rate = config["PerfTunning"]["BenchMarkServing"]["max_request_rate"]
            grid_request_rate = config["PerfTunning"]["BenchMarkServing"]["grid_request_rate"]
            min_num_prompts = config["PerfTunning"]["BenchMarkServing"]["min_num_prompts"]
            max_num_prompts = config["PerfTunning"]["BenchMarkServing"]["max_num_prompts"]
            grid_num_prompts = config["PerfTunning"]["BenchMarkServing"]["grid_num_prompts"]
            results = {}
            results["perf_results"] = {}
            all_choices = [("--enable-chunked-prefill ", enable_chunked_prefill), ("--enable-prefix-caching ",enable_prefix_caching), ("--disable-async-output-proc ",disable_async_output_proc), ("--enforce-eager", enforce_eager)]
            choices = []
            for i in all_choices:
                if i[1]:
                    choices.append(i[0])
            serving_cmd = f"python3 -m vllm.entrypoints.openai.api_server \
                           --model {model_path} \
                           --dtype {dtype} \
                           --device={device} \
                           --block-size {block_size} " + \
                           "--trust_remote_code " * trust_remote_code + \
                           f"--max-model-len {max_model_len} \
                           --pipeline-parallel-size {pipeline_parallel_size} \
                           --tensor-parallel-size {tensor_parallel_size} \
                           --pipeline-parallel-size {pipeline_parallel_size} " + \
                           f"--distributed-executor-backend {distributed_executor_backend}" * (distributed_executor_backend!=None) + \
                           f"--gpu-memory-utilization {gpu_memory_utilization} \
                           --max-num-seqs {max_num_seqs} \
                           --quantization {quantization} "

            # find best choices of EngineArgs
            best_throughput = -1
            best_request_rate = -1
            best_num_prompt = -1
            selected_choices = []
            best_serving_cmd = best_client_cmd = ""
            for choice in choices:
                serving_cmd += choice
                logging.info("run serving cmd: %s", serving_cmd)
                print("run serving cmd: %s", serving_cmd)
                def server_func():
                    os.system(serving_cmd)
                t1 = threading.Thread(target=server_func)
                t1.start()
                #sleep(SLEEP_TIME)

                if not is_server_ready():
                    print("serving failed...")
                    logging.info("serving failed...")
                    return

                cmd = f"python3 -m vllm_utils.benchmark_serving \
                        --backend vllm \
                        --model {model_path} \
                        --dataset-name {dataset_name} " + \
                       "--dataset-path {dataset_path}" * (dataset_path!=None) + \
                       f"--request-rate inf \
                       --num-prompts 10 " + \
                       f"--max-concurrency {max_concurrency} " * (max_concurrency!=None) + \
                       f"--{dataset_name}-input-len {input_len} \
                       --{dataset_name}-output-len {output_len} " + \
                       "--ignore_eos " * ignore_eos \
                       + "--strict-in-out-len " * strict_in_out_len + \
                       "--keep_special_tokens " * keep_special_tokens + \
                       "--trust-remote_code " * trust_remote_code                
                def client_func():
                    output_str = os.popen(cmd).read()
                    return output_str
                t2 = ThreadWithReturnValue(target=client_func)
                t2.start()
                output_str = t2.join()
                start = [i.start() for i in re.finditer('{', output_str)][-1]
                end = [i.start() for i in re.finditer('}', output_str)][-1]
                perf_result = json.loads(output_str[start:(end+1)])
                TTFT, TPOT, out_token_tp = perf_result["ttft"], perf_result["tpot"], perf_result["out_tp"]
                if TTFT >= required_ttft and TPOT >= required_tpot:
                    serving_cmd.replace(choice, "")
                    continue
                if out_token_tp >= best_throughput:
                    best_serving_cmd = serving_cmd
                    selected_choices.append(choice)
                    logging.info("find better choices: %s", selected_choices)
                    best_throughput = out_token_tp
                else:
                    serving_cmd.replace(choice, "")
                
                os.system("ps -ef | grep vllm| awk '{print $2}' | xargs kill -9")
                os.system("ps -ef | grep multiprocessing| awk '{print $2}' | xargs kill -9")
                torch.gcu.empty_cache()

            logging.info("Selected_choices: %s", str(selected_choices))
            print("Selected_choices: %s", str(selected_choices))
            
            def server_func():
                os.system(best_serving_cmd)
            t1 = threading.Thread(target=server_func)
            t1.start()
            #sleep(SLEEP_TIME)

            if not is_server_ready():
                print("serving failed...")
                logging.info("serving failed...")
                return
            
            # find best combination of request_rate and num_prompts
            request_rate_list = []
            rr = 0.1
            while rr <= max_request_rate:
                rr += grid_request_rate
                request_rate_list.append(rr)
            logging.info("request_rate_list: %s", str(request_rate_list))
            logging.info("num_propmpts_list: %s", str(list(range(min_num_prompts, max_num_prompts+grid_num_prompts, grid_num_prompts))))
            print("request_rate_list: %s", str(request_rate_list))
            print("num_propmpts_list: %s", str(list(range(min_num_prompts, max_num_prompts+grid_num_prompts, grid_num_prompts))))
            for request_rate in request_rate_list + ["inf"]:
                for num_prompts in range(min_num_prompts, max_num_prompts+grid_num_prompts, grid_num_prompts):
                    cmd = f"python3 -m vllm_utils.benchmark_serving \
                            --backend vllm \
                            --model {model_path} \
                            --dataset-name {dataset_name} " + \
                            "--dataset-path {dataset_path}" * (dataset_path!=None) + \
                            f"--request-rate {request_rate} \
                            --num-prompts {num_prompts} " + \
                            f"--max-concurrency {max_concurrency}"*(max_concurrency!=None) + \
                            f"--{dataset_name}-input-len {input_len} \
                            --{dataset_name}-output-len {output_len} " + \
                            "--ignore_eos " * ignore_eos + \
                            "--strict-in-out-len " * strict_in_out_len + \
                            "--keep_special_tokens " * keep_special_tokens + \
                            "--trust-remote_code " * trust_remote_code
                    logging.info("run command: %s", cmd)
                    print("run command: %s", cmd)
                    def client_func():
                        output_str = os.popen(cmd).read()
                        return output_str
                    t2 = ThreadWithReturnValue(target=client_func)
                    t2.start()
                    output_str = t2.join()
                    start = [i.start() for i in re.finditer('{', output_str)][-1]
                    end = [i.start() for i in re.finditer('}', output_str)][-1]
                    perf_result = json.loads(output_str[start:(end+1)])
                    TTFT, TPOT, out_token_tp = perf_result["ttft"], perf_result["tpot"], perf_result["out_tp"]

                    if TTFT >= required_ttft and TPOT >= required_tpot:
                        continue
                    if out_token_tp >= best_throughput:
                        logging.info("find better (request_rate, num_prompts): %s", str((request_rate,num_prompts)))
                        print("find better (request_rate, num_prompts): %s", str((request_rate,num_prompts)))
                        best_throughput = out_token_tp
                        best_client_cmd = cmd
                    else:
                        logging.info("skip request_rate: " + str(request_rate) + " num_prompts: " + str(num_prompts))
                        print("skip request_rate: " + str(request_rate) + " num_prompts: " + str(num_prompts))
            
            logging.info("*** Final Result ***")
            logging.info("best_serving_cmd: %s", best_serving_cmd)
            logging.info("best_client_cmd: %s", best_client_cmd)
            logging.info("best_throughput: %s", best_throughput)
            print("*** Final Result ***")
            print("best_serving_cmd: %s", best_serving_cmd)
            print("best_client_cmd: %s", best_client_cmd)
            print("best_throughput: %s", best_throughput)
    
    logging.info(f"Finished running {MetricType} using {EvalTool}")
    print(f"Finished running {MetricType} using {EvalTool}")

if __name__ == '__main__':
    main()

Intro:  
A toolkit to evaluate and find the best combination of LLM parameters.   
One only needs to edit the config file to run [Perf, Acc, PerfTunning, PerfAnalysis].  

Usage:  
python3 run.py --config ./config/config.yaml  

Must set the following items in the config, other items can be filled as needed:  
1. Model: model_name, model_path, tokenizer_path.  
2. MetricType: select from [Perf, Acc, PerfTunning, PerfAnalysis].  
3. EvalTool: select from [BenchmarkTest, BenchmarkServing, OpenCompass, lm_eval, EvalScope].  
4. InferType: select from [offline, serving].  

Functions:
1. Perf: BenchmarkTest, BenchmarkServing, Evalscope
2. Acc: OpenCompass, BenchmarkTest, lm_eval
3. PerfTunning:  
(1) BenchmarkTest(tunning params: num_prompts + EngineArgs).  
(2) BenchmarkServing (tunning params: request_rate, num_prompts + EngineArgs).  

Notes:  
1. PerfTunning only supports grid searching with given min max. Possibly add other methods (bisection...). 

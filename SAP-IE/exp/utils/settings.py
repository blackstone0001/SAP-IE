# jc 23.10.30

import os
import sys
import torch
import argparse

from transformers import GenerationConfig, BitsAndBytesConfig

LLM_PATH = "/home/jc/workspace/llama/Llama-2-13b-hf/"
TOKENIZER_PATH = "/home/jc/workspace/llama/Llama-2-13b-hf/"
model_path = {
    'llama-7b': "/date/jc/models/llama/Llama-2-7b-hf/",
    'llama-13b': "/date/jc/models/llama/Llama-2-13b-hf/",
    'code-llama-13b-python': "/date/jc/models/llama/CodeLlama-13b-Python-hf/",
    'code-llama-34b-python': "/date/jc/models/llama/CodeLlama-34b-Python-hf/",
    'code-llama-13b-instruct': "/date/jc/models/llama/CodeLlama-13b-Instruct-hf/",
    'code-llama-7b-python': "/date/jc/models/llama/CodeLlama-7b-Python-hf/",
    'mistral-7b': '/date/jc/models/models/Mistral-7B-v0.1',
    'gemma-7b': '/date/jc/models/models/gemma-7b',
    'qwen-7b': '/date/jc/models/Qwen1.5-7B',
    'gpt-3.5-turbo-7b': '',
}

GENERATION_CONFIG = GenerationConfig(
    temperature=0.5,
    # top_k=40,
    # top_p=0.9,
    do_sample=False,
    # num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=400
    )
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

def parse_args():
    parser = argparse.ArgumentParser()

    # For model/ setup
    parser.add_argument('--use_api', default=False, action='store_true',help="Use the API for inference")
    parser.add_argument('--model_name', default="mistral-7b", type=str)
    # parser.add_argument('--base_model', default=LLM_PATH, type=str)
    parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
    # parser.add_argument('--tokenizer_path',default=TOKENIZER_PATH,type=str)
    parser.add_argument('--data_file', default=None, type=str,help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--out_dir', default="/home/jc/workspace/exp/test/test_log", type=str,help="Output directory of test log files.")
    parser.add_argument('--out_file_suffix', default="", type=str,help="Suffix of the output file name.")
    parser.add_argument('--interactive', default=True, action='store_true',help="run in the instruction mode (single-turn)")
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true',help='only use CPU for inference')
    parser.add_argument('--alpha', type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
    parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
    
    # For test
    parser.add_argument('--metric_type', default='f1', choices=['accuracy', 'f1'], type=str, help="The metric type for evaluation")
    parser.add_argument('--eval_only', default=False, action='store_true', help="Only evaluate the result without generating it again") # deprectead
    parser.add_argument('--test_line_num', default=10, type=int)
    parser.add_argument('--shot', default=5, type=int)
    parser.add_argument('--dataset', default='conll2003', type=str)
    parser.add_argument('--task', default='code-ner', type=str)

    # For prompt
    parser.add_argument('--with_prompt', default=False, action='store_true',help="Wrap the input with the text prompt template automatically")
    parser.add_argument('--prompt_style', default='python', choices=['python', 'natural'], type=str)
    parser.add_argument('--prompt_strat', default='type-aware', choices=['random', 'type-aware'], type=str)

    # For knowledge base
    parser.add_argument('--kb', default=False, action='store_true', help="Use knowledge base")
    parser.add_argument('--kb_path', default="/home/jc/workspace/exp/kbase/task_template.csv", help="Path of the KB")
    parser.add_argument('--kb_thres', default=5, type=int, help="Threshold of the similarity between the input sentence and the KB sentence")

    parser.add_argument('--check_fact', default=False, action='store_true', help="Use fact checking module")
    parser.add_argument('--out_file_to_check_path', default="", help="Path of the file to check fact")
    parser.add_argument('--checked_out_dir', default="/home/jc/workspace/exp/figs/data/checked", help="Output dir path of the checked file")

    # Generation Config -- all are default
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--top_p', default=1, type=float)
    parser.add_argument('--do_sample', default=False, action='store_true')
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--repetition_penalty', default=1.0, type=float)
    parser.add_argument('--max_new_tokens', default=300, type=int)

    args = parser.parse_args()
    return args

def apply_settings(args):
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    return device

    # apply_attention_patch(use_memory_efficient_attention=True)
    # apply_ntk_scaling_patch(args.alpha)

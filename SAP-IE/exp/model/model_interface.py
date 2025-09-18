# jc 23.10.30

import json
from shutil import copyfile

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from exp.utils.settings import *
from exp.model.model_api import *

def gen_prompt(instruction, input=None, template=PROMPT_TEMPLATE):
    """Generate prompt before feeding inputs into LLM"""
    if input:
        instruction = instruction + '\n' + input
    return template.format_map({'instruction': instruction})

class ModelInterface:
    def __init__(self):
        self.args = None
        self.generation_config = None
        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.setup()
    
    def setup(self):
        self.args = parse_args()
        self.device = apply_settings(args=self.args)
        self.model_path = model_path[self.args.model_name]

        self.generation_config = GenerationConfig(
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=self.args.do_sample,
            num_beams=self.args.num_beams,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_new_tokens=self.args.max_new_tokens,
        )
        if not self.args.use_api:
            self.load_model()

    def load_model(self):
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,     # 用4bit动态量化，大大减少资源需求
        #     # bnb_4bit_use_double_quant=True,
        #     # bnb_4bit_quant_type="nf4",
        #     # bnb_4bit_compute_dtype=torch.bfloat16
        # )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map='auto',  # 使模型能够移动到GPU
            load_in_4bit=True
            # quantization_config=bnb_config
            )
        
        # 检查模型字典词数
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f">>> Model name: {self.args.model_name}")
        print(f">>> Vocab of the base model: {model_vocab_size}")
        print(f">>> Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size!=tokenzier_vocab_size and self.args.model_name != 'qwen-7b':
            assert tokenzier_vocab_size > model_vocab_size  # 训练字典集中的词数不能小于测试集中的词数，否则会超纲
            print(">>> Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)
        
        # 合并微调模型
        if self.args.lora_model is not None:
            print(">>> Loading peft model")
            load_type = torch.float16
            model = PeftModel.from_pretrained(
                base_model, 
                self.args.lora_model,
                torch_dtype=load_type,
                device_map='auto',)
        else:
            model = base_model

        if self.device==torch.device('cpu'):
            model.float()   # 将模型转换为float32，以便在CPU上运行

        model.eval()    # 使模型进入 evaluation mode
        print(">>> Finish loading model and tokenizer")

        self.model, self.tokenizer = model, tokenizer

    def run_iter(self):
        """Run LLM iteratively"""
        print(">>> Running LLM iteratively")
        with torch.no_grad():
            while True:
                raw_input_text = input(">>> Input: ")
                if len(raw_input_text.strip())==0:
                    break

                if self.args.with_prompt:
                    input_text = gen_prompt(instruction=raw_input_text)
                else:
                    input_text = raw_input_text

                model_inputs = self.tokenizer(input_text, return_tensors="pt")
                generated_ids = self.model.generate( 
                    input_ids = model_inputs["input_ids"].to(self.device),
                    attention_mask = model_inputs['attention_mask'].to(self.device),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    generation_config=self.generation_config
                    )[0]
                output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                if self.args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output
                print(">>> Response: ", response, "\n")

    def run_batch(self, data, file_out, with_prompt):
        print(">>> Feeding data to LLM in a batch")
        results = []

        if self.args.use_api:
            print(">>> Using API for inference")
            for index, example in enumerate(data):
                if with_prompt is True:
                    input_text = gen_prompt(instruction=example)
                else:
                    input_text = example
                response = generate_text(input_text)
                print(f"======={index}=======")
                print(f"Input: \n{example}\n")
                # print(f"Output: \n{input_text+response}\n")
                print(f"Output: \n{response}\n")
                results.append({"Input":input_text, "Output":response})
        else:
            with torch.no_grad():
                for index, example in enumerate(data):
                    if with_prompt is True:
                        input_text = gen_prompt(instruction=example)
                    else:
                        input_text = example
                    inputs = self.tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                    # print(len(inputs["input_ids"][0]))
                    generated_ids = self.model.generate(
                        input_ids = inputs["input_ids"].to(self.device),
                        attention_mask = inputs['attention_mask'].to(self.device),
                        # eos_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=2,
                        # pad_token_id=self.tokenizer.pad_token_id,
                        pad_token_id=2,
                        generation_config=self.generation_config
                        )[0]
                    output = self.tokenizer.decode(generated_ids,skip_special_tokens=True)
                    if with_prompt:
                        response = output.split("### Response:")[1].strip()
                    else:
                        response = output
                    print(f"======={index}=======")
                    print(f"Input: \n{example}\n")
                    print(f"Output: \n{response}\n")

                    results.append({"Input":input_text, "Output":response})

        if file_out:
            if os.path.exists(file_out):
                copyfile(file_out, file_out+".bak")
            with open(file_out, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            return results

if __name__ == '__main__':
    mi = ModelInterface()
    mi.setup()

    data = [
        "Please tell me a joke.",
        "Please write a slogan",
    ]
    file_out = "/home/jc/test.json"
    # mi.run_iter()
    mi.run_batch(data, file_out, with_prompt=True)

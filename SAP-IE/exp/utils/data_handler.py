# jc 23.10.30

import json
import os
import random
import re
import sys
import traceback
import pandas as pd
import ast

# os.chdir('./exp/')

TEST_DATAFILE = {
    'scierc': "sciERC/processed_data/test.json",
    'nyt': "nyt10/processed_data/nyt10_test.txt",
    'tacred': "tacred/test.txt",
    'webnlg': "webnlg/en/processed_data/test.json",
    'conll2003': 'conll2003/processed_data/test.json',
    'wikigold': 'wikigold/processed_data/test.json',
    # 'wikigold': 'Wiki/processed_data/test.json',
    'ontonotes': 'OntoNotes/processed_data/test.json',
    # 'ontonotes': 'OntoNote/processed_data/test.json',
    'bbn': 'BBN/processed_data/test.json',
    'ace2005ner': 'ace2005/ner/processed_data/test.json',
    'ace2005re': 'ace2005/re/processed_data/test.json',
    'conll2004': 'conll2004/processed_data/test.json',
    }

class DataHandler(object):

    def __init__(self, dataset, task, line_num=-1, data_file=None,
                 DATA_DIR="/home/jc/workspace/exp/data", 
                 FILE_OUT_DIR = "/home/jc/workspace/exp/test/test_log"
                 ):
        self.DATA_DIR = DATA_DIR
        self.FILE_OUT_DIR = FILE_OUT_DIR
        self.df = None  # Dataframe
        self.dataset = dataset
        self.task = task
        self.line_num = line_num    # if -1, use all lines
        if data_file:
            self.file_in = data_file
        else:
            self.file_in = os.path.join(self.DATA_DIR, TEST_DATAFILE[self.dataset])
        self.file_out = None

    def set_output_path(self, file_out):
        self.file_out = file_out

    @staticmethod
    def read_json_line_file(file_in)->list:
        """
        Read files like:
        {}  # without comma
        {}
        """
        result = []
        with open(file_in, 'r') as f:
            line = f.readlines()
            result = [json.loads(i) for i in line]
            f.close()
        return result
    
    @staticmethod
    def read_json_file(file_in):
        result = []
        with open(file_in, 'r') as f:
            result = json.load(f)
            f.close()
        return result

    @staticmethod
    def write_data_file(file_out, *content):
        """
        Write data to a file.

        Args:
            file_out (str): The path of the output file.
            content (list): The list of data to be written to the file. 
                            Each element in this tuple is a direct list of a line to be write.
                            Example: content = (list1, list2, ...)
                            list1 = [line1, line2, ...]
                            list1 can not be [[line1, line2, ...]]

        Returns:
            None
        """
        with open(file_out, 'w') as f:
            for block in content:
                for item in block:
                    if type(item) == dict:
                        string = json.dumps(item)
                    else:
                        string = str(item)
                    f.write(string)
                    f.write("\n")
            f.close()

    def load_random_test_data(self):
        """
        Load test data from file to a unified format:
        NER: {'sentence': str, 'ner': list[(str, str)]}
        RE: {'sentence': str, 're': list[(str, str, str)]}
        """
        if self.dataset in ['nyt', 'tacred', 'ace2005re', 'conll2004']:
            self._load_test_data_re()
        elif self.dataset in ['conll2003', 'wikigold', 'ontonotes', 'bbn', 'ace2005ner']:
            self._load_test_data_ner()
        elif self.dataset in ['scierc']:
            self._load_test_data_ner_re()
        elif self.dataset == 'webnlg':
            self._load_test_data_webnlg()
        else:
            raise NotImplementedError("Dataset {} not implemented to load_test_data.".format(self.dataset))

        if self.dataset in ['nyt', 'tacred']:   # Exclude NA relations
            if self.dataset in ['nyt', 'tacred']:
                mask = self.df['re'].apply(lambda x: x[0][1] != 'NA')
            # elif self.dataset == 'ace2005re':
            #     mask = self.df['re'].apply(lambda x: len(x) != 0)

            if self.line_num != -1:
                    self.df = self.df[mask].sample(self.line_num, random_state=random.randint(1, 50), axis=0, replace=True)
            else:
                self.df = self.df[mask]
            return

        if self.line_num != -1:
            self.df = self.df.sample(self.line_num, random_state=random.randint(1, 50), axis=0)

    def _load_test_data_ner(self):
        data = self.read_json_line_file(self.file_in)
        result = []
        for item in data:
            tmp = {}
            tmp['sentence'] = item['sentence'].replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']').replace('{', '-LCB-').replace('}', '-RCB-')
            tmp['ner'] = item['ner']
            tmp['ner'] = [(dic['name'], dic['type']) for dic in item['ner']]
            result.append(tmp)
        self.df = pd.DataFrame(result)

    def _load_test_data_re(self):
        data = self.read_json_line_file(self.file_in)
        result = []
        for item in data:
            tmp = {}
            if self.dataset == 'tacred':
                tmp['sentence'] = " ".join(item['token']).replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']').replace('{', '-LCB-').replace('}', '-RCB-')
                tmp['re'] = [(item['h']['name'], item['relation'], item['t']['name'])]
            elif self.dataset == 'nyt':
                tmp['sentence'] = item['text'].replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']').replace('{', '-LCB-').replace('}', '-RCB-')
                tmp['re'] = [(item['h']['name'], item['relation'], item['t']['name'])]
            elif self.dataset in ['ace2005re', 'conll2004']:
                tmp['sentence'] = item['sentence'].replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']').replace('{', '-LCB-').replace('}', '-RCB-')
                tmp['re'] = [(dic['h']['name'], dic['relation'], dic['t']['name']) for dic in item['re']]
            result.append(tmp)
        self.df = pd.DataFrame(result)

    def _load_test_data_ner_re(self):
        """Load data for both NER and RE tasks"""
        data = self.read_json_line_file(self.file_in)
        result = []
        for item in data:
            tmp = {}
            tmp['sentence'] = item['sentence']
            tmp['ner'] = [(dic['name'], dic['type']) for dic in item['ner']]
            tmp['re'] = [(dic['h'], dic['relation'], dic['t']) for dic in item['re']]
            result.append(tmp)
        self.df =  pd.DataFrame(result)

    def _load_test_data_webnlg(self):
        data = self.read_json_file(self.file_in)['entries']
        result = []
        for index in range(len(data)):
            item = data[index][str(index+1)]
            tmp = {}
            tmp['sentence'] = item['lexicalisations'][0]['lex'].replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']')
            tmp['relation'] = item['modifiedtripleset']
            for i, j in enumerate(tmp['relation']):
                tmp['relation'][i] = [j['subject'], j['property'], j['object']]
            result.append(tmp)
        self.df = pd.DataFrame(result)

    def prepare_eval_data(self, prompt_style, use_api=False):
        return self._prepare_test(prompt_style, use_api)
        # else:
        #     raise NotImplementedError("Task {} not implemented to prepare_eval_data.".format(self.dataset))

    def _prepare_test(self, prompt_style, use_api):
        """
        target: Dataframe, including sentence and label
        pred: Including only predicted label
        """
        target = self.df
        sentences = []
        pred = []
        with open(self.file_out, 'r') as f:
            s = json.load(f)
            for index, item in enumerate(s):
                try:
                    if use_api:
                        input_sentence, tmp = handle_api_output(item, self.task)
                    else:
                        if prompt_style == 'natural':
                            input_sentence, tmp = handle_natural_output(item, self.task, "The given sentence is: ", "### Extract", "Extracted")
                        elif prompt_style == 'python':
                            input_sentence, tmp = handle_python_output(item, self.task, "input_text = ", "def", None)
                except Exception as e:
                    # print("Error when evaling output line ", index, ": ", e)
                    tmp = []
                    input_sentence = ""
                else:
                    pass
                sentences.append(input_sentence)
                pred.append(tmp)
        return target, sentences, pred
    
    def prepare_check_fact_data(self, output_dict, prompt_style, sentence_head="The given sentence is:", shot_head="### Rate", answer_head="Rate score", match_str=r'"(\d+)"'):
        pred = []
        for index, item in enumerate(output_dict):
            try:
                if prompt_style == 'natural':
                    input_sentence, tmp = handle_natural_output(item, self.task, sentence_head, shot_head, answer_head, match_str=match_str)
                elif prompt_style == 'python':
                    input_sentence, tmp = handle_python_output(item, self.task)
            except Exception as e:
                # print("Error when evaling output line ", index, ": ", e)
                tmp = []
            else: pass
            pred.append(tmp)
        return pred

    def load_output_file(self, file_path, comlumns=['sentence', 'target']):
        """
        Load output file and return metrics, target and prediction.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if line.startswith("Metric"):
                nums = line.split(":")[1][1:-2].strip()
                metrics_before_check = eval(nums)
            if line.startswith("target"):
                break
        lines = lines[i+1:]

        sentence, target, prediction = [], [], []
        for i in range(0, len(lines), 4):
            sentence.append(lines[i].strip())
            target.append(eval(lines[i+1].strip()))
            prediction.append(eval(lines[i+2].strip()))
        target_df = pd.DataFrame([sentence, target]).T
        target_df.columns = comlumns
        return metrics_before_check, target_df, prediction

def handle_natural_output(item, task, sentence_head, shot_head, answer_head, match_str=r'\[.*?\]'):
    """
    Example:
    "### Extract the entities ....
        The given sentence is: I am a student.
        Extracted result: [('student', 'PER')]"
    sentence_head: "The given sentence is: "
    shot_head: "### Extract"
    answer_head: "Extracted"
    """
    try:
        in_str = item['Input']
        tmp = item['Output'].replace('“', '"').replace('”', '"')    # 替换中文符号
        
        in_div_string = sentence_head
        in_matches = re.finditer(in_div_string, in_str)     # find all matches in the input
        for i in in_matches:        # make i = index of the last match
            in_index = i.start()
        in_str = in_str[in_index:]      # split and get the section we need
        input_sentence = re.findall(r'\"(.*?)\"', in_str)[0]

        list_string = extract_substring(input_sentence, tmp, shot_head, answer_head)
        print(">>> list_string1: ", list_string)
        list_string = re.findall(match_str, list_string)
        print(">>> list_string2: ", list_string)
        try:
            lst = [ast.literal_eval(ds) for ds in list_string][0]
            print(">>> lst: ", lst)
        except Exception as e:
            print(">>> Error when evaling natural output: ", e)
            print(">>> list_string: ", list_string)
            raise e
        return input_sentence, lst
    except Exception as e:
        traceback.print_exc()
        print("Error when evaling output: ", e)
        raise e

def handle_python_output(item, task, sentence_head, shot_head, answer_head):
    try:
        in_str = item['Input']
        out_str = item['Output'].replace('“', '"').replace('”', '"')    # 替换中文符号
        
        in_div_string = sentence_head
        in_matches = re.finditer(in_div_string, in_str)     # find all matches in the input
        for i in in_matches:        # make i = index of the last match
            in_index = i.start()
        in_str = in_str[in_index:]      # split and get the section we need
        input_sentence = re.findall(r'\"(.*?)\"', in_str)[0]
        
        dict_strings = extract_substring(input_sentence, out_str, shot_head)
        dict_strings = re.findall(r'\{.*?\}', dict_strings)
        # 使用 ast.literal_eval 将字符串转换为字典
        dicts = [ast.literal_eval(ds) for ds in dict_strings]

        for i, j in enumerate(dicts):
            if task == 'ner':
                dicts[i] = (j['text'], j['type'])
            elif task in ['re', 'pure-re']:
                dicts[i] = (j['ent1_text'], j['rel_type'], j['ent2_text'])

        return input_sentence, dicts
    except Exception as e:
        print("Error when evaling output: ", e)
        raise e

def handle_api_output(item, task):
    if task == 'ner':
        map_str = r'\[.*?\]'
    else:
        map_str = r'\(.*?\)'
    list_string = re.findall(map_str, item['Output'])
    print(">>> list_string: ", list_string)
    try:
        if task == 'ner':
            lst = [ast.literal_eval(ds) for ds in list_string][0]
        else:
            lst = [ast.literal_eval(ds) for ds in list_string]
        print(">>> lst: ", lst)
    except Exception as e:
        print(">>> Error when evaling natural output: ", e)
        print(">>> list_string: ", list_string)
        raise e
    return "", lst

def extract_substring(input_sentence, string, shot_head, answer_head=None):
    """
    从string中提取对应的答案: 在string中,找到包含 input_sentence 的位置，并找出这个位置之后可能包含 "shot_head"(such as: "def") 字符串的位置（也可能不包含这个字符串，此时就返回字符串的结尾位置），最后取出这两个位置之间的字符串。
    return: 从 input_sentence 到 shot_head 的子字符串, 如果 "shot_head"(such as: "def") 不存在，则从字符串的结尾到 input_sentence 的子字符串。如果 input_sentence 不存在于 out_str 中，返回 None
    """
    start = string.find(input_sentence)
    if start == -1:
        return None
    start += len(input_sentence) + 1    # 1 is the length of the double quotation marks
    end = string[start:].find(shot_head) + start
    if end == -1:
        end = len(string)
    if not answer_head:  # start_mark: like 'Extracted', to identify the given entity list and the answer list in pure-re task. Because both of them are wrapped by []
        return string[start:end]    
    else:
        sub = string[start:end]
        new_start = sub.find(answer_head)
        return sub[new_start:]

def load_checked(file_path, mode='check'):
    """Load checked log file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith("Metric"):
            nums = line.split(":")[1][1:-2].strip()
            metrics_before_check = eval(nums)
        if line.startswith("target"):
            break
    lines = lines[i+1:]

    sentence, target, prediction, checked = [], [], [], []
    if mode == 'check':
        step = 5
    else:
        step = 4
    for i in range(0, len(lines), step):
        sentence.append(lines[i].strip())
        target.append(eval(lines[i+1].strip()))
        prediction.append(eval(lines[i+2].strip()))
        if mode == 'check':
            checked.append(eval(lines[i+3].strip()))
    return sentence, target, prediction, checked, metrics_before_check
    

if __name__ == '__main__':
    # dh = DataHandler('conll2003', 'ner', line_num=3)
    # m, target, pred = dh.load_output_file("/home/jc/workspace/exp/test/test_log/conll2003_code-ner_llama-13b_shot-3_maxline-50_rpenal-1.0_maxtoken-300_beam-1_temp-0.1_k-5_p-0.1.json.log", ['sentence', 'ner'])
    # print("metrics: ", m)
    # print("target: ", target)
    # print(target['ner'][0])
    # print("pred: ", pred)

    dh = DataHandler('scierc', 'ner', line_num=3)

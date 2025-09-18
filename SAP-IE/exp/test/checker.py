# jc 24.3.10

import random

from exp.prompt.template import *
from exp.utils.data_handler import DataHandler
from exp.utils.kb_handler import *
from exp.test.prompts import *

task_map = {
    'ner': 'ner',
    're': 're',
    'pure-re': 're',
}


def read_options():
    with open('/home/jc/workspace/exp/prompt/type_dict_of_test.json', 'r') as f:
        dic = json.load(f)
        options = {}
        for dataset, value in dic.items():
            options[dataset] = {}
            for task, value_2 in value.items():
                options[dataset][task] = list(value_2.keys())
                try:
                    options[dataset][task].remove('NA')
                except: pass
                    # print("No 'NA' in the options")
    return options
Type_Options = read_options()

prompt_head = (
    "Task: Extract all the entities and their types in <The given sentence> according to the <Type Options>, in the form of (<name>, <type>)\n"
    "<Type Options>: {type_options}\n"
    "<RULE>\n"
    "1. The <name> is in <The given sentence>\n"
    "2. The <type> is in <Type Options>\n"
    "3. The <name> and the corresponding <type> are reasonable\n"
    # "4. The <name> and <type> can be inferede from <The given sentence> (2 points)\n"
    # "5. <The answer> not miss any named entity mentioned in <The given sentence> (2 points)\n"
    "</RULE>\n"
    "Please score <The answer> according to <RULE> (0~10, higher the better) :\n"
)
prompt_template = (
    # "### Task: Extract all the entities and their types\n"
    "### Rate the result of Name Entity Recognition according to its reliability\n"
    "\tThe given sentence is: \"{sentence}\"\n"
    "\tExtracted result: {answer}\n"
    # "Score: {score}\n"
    "\tRate score: {score}\n"
)


class Checker:
    def __init__(self, dataset, task, shot, file_path=None,
                 kb_path=None, prompt_style='natural'):
        self.dataset = dataset
        self.task = task
        self.shot = shot
        self.file_path = file_path  # path of the file to be checked

        self.dh = DataHandler(dataset, task)
        self.kb = KBHandler(task, shot, kb_path=kb_path, mode='check') if kb_path else None
        self.prompt_style = prompt_style

        self.prompt_head = revise_prompts[task_map[task]]['head']
        self.prompt_tail = revise_prompts[task_map[task]]['tail']

    def annotate(self):
        self.metric, self.target, self.pred = self.dh.load_output_file(self.file_path, comlumns=['sentence', task_map[self.task]])
        full = self.kb.score_kb_full()
        if full <= 0:
            return
        
        sentence = list(self.target['sentence'])
        label = list(self.target[task_map[self.task]])
        pred = self.pred
        rand_indexes = random.sample(range(len(sentence)), min(full, len(sentence)))
        # sent_list, pred_list, score_list = list(sentence), list(pred), list(label)
        sent_list, pred_list, score_list = [], [], []
        for i in rand_indexes:
            sent_list.append(sentence[i])
            pred_list.append(pred[i])
            score_list.append(label[i])
        self.kb.update_revise_kb(sent_list, pred_list, score_list)

        # self.collect_score(sentence, label, full)

    def collect_score(self, sentence, label, full):
        sent_list, pred_list, score_list = [], [], []
        for i in random.sample(range(len(sentence)), full):
            if len(set(label[i]) & set(self.pred[i])) == len(label[i]):
                score = 10
            elif len(self.pred[i]) == 0 and len(label[i]) != 0:
                score = 0
            elif len(self.pred[i]) != 0 and len(label[i]) == 0:
                score = 0
            # elif len(set(self.pred[i]) & set(label[i])) == 0:
            #     score = 0
            # elif (len(set(self.pred[i]) & set(label[i])) == 0) and (not (self.dataset == 'nyt' or self.dataset == 'tacred')):
            #     score = 0
            else:
                print("\n\n>>> Sentence: ", sentence[i], "\n")
                print("Label: ", label[i], "\n")
                print("Prediction: ", self.pred[i], "\n")
                print("Len(pred): ", len(self.pred[i]))
                print("true: ", len(set(label[i]) & set(self.pred[i])))
                try:
                    score = eval(input("Please give a score for the prediction: (0~10, higher the better)"))
                except Exception as e:
                    print(e)
                    continue
            sent_list.append(sentence[i])
            pred_list.append(self.pred[i])
            score_list.append(score)
        self.kb.update_score_kb(sent_list, pred_list, score_list)

    def get_check_prompt(self):#, input_sentence):
        shots = []
        samples = self.kb.get_answer_score()
        if self.prompt_style == 'python':
            shots = self._build_python_shots(samples)
        elif self.prompt_style == 'natural':
            # shots = self._build_natural_shots(samples)
            shots = self._build_natural_revise_shots(samples)
        # return "".join(shots) + prompt_template[:-8]
        # return "".join(shots) + prompt_template[0:find_second_last_occurance(prompt_template, '\n')] + "\n"
        return "".join(shots) + self.prompt_tail[0:find_second_last_occurance(self.prompt_tail, '\n')] + "\n"

    def _build_natural_shots(self, samples):
        """Build natural language style shots from samples"""
        # shots = [prompt_head.format_map({'type_options': str(Type_Options[self.dataset])})]
        shots = []
        for i in range(samples.shape[0]):
            tmp_shot = prompt_template
            sample = samples.iloc[i]
            sentence = sample['sentence']
            pred = sample['pred']
            score = sample['score']
            tmp_shot = tmp_shot.format_map({'sentence': sentence, 'answer': str(pred), 'score': "\"" + str(score) + "\""})
            shots.append(tmp_shot)
            # shots.append(prompt_head.format_map({'type_options': str(Type_Options[self.dataset])}))   ### new
        return shots

    def _build_natural_revise_shots(self, samples):
        shots = [self.prompt_head.format_map({'type_options': str(Type_Options[self.dataset][task_map[self.task]])})]
        # shots = []
        for i in range(samples.shape[0]):
            tmp_shot = self.prompt_tail
            sample = samples.iloc[i]
            sentence = sample['sentence']
            pred = sample['pred']
            label = sample['label']
            tmp_shot = tmp_shot.format_map({'sentence': sentence, 'answer': str(pred), 'label': str(label)})
            shots.append(tmp_shot)
            # shots.append(prompt_head.format_map({'type_options': str(Type_Options[self.dataset])}))   ### new
        return shots

def find_second_last_occurance(str, substr):
    return str.rfind(substr, 0, str.rfind(substr))

if __name__ == '__main__':
    t = read_options()
    print(t)
    # print(len(prompt_head) + len(prompt_template))

    # dataset = 'conll2003'
    # task = 'ner'
    # shot = 25
    # ck = Checker(dataset, task, shot, 
    #              file_path='/home/jc/workspace/exp/test/test_log/conll2003_ner_natural_llama-7b_shot-25_maxline-50_rpenal-1.0_maxtoken-300_beam-1.json.4.log',
    #              kb_path=f'/home/jc/workspace/exp/kbase/check_score_{dataset}.csv')
    # ck.annotate()
    # # metric, target, pred = ck.dh.load_output_file(ck.file_path, comlumns=['sentence', task_map[ck.task]])
    # sentence = ck.target['sentence']

    # inputs = []
    # for i in range(len(sentence)):
        # print(f"{i}-th line: ", ck.get_check_prompt(sentence[i]))
    # input_str = ck.get_check_prompt(sentence[i]).format_map({'sentence': sentence[i], 'answer': str(ck.pred[i])})
    # inputs.append(input_str)
    # print(inputs[0])

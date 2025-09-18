
from collections import defaultdict
import os
import random
import sys
import json

sys.path.append('/home/jc/workspace/')
from exp.utils.data_handler import DataHandler
from exp.prompt.template import *
from exp.utils.kb_handler import *

task_map = {
    'ner': 'ner',
    're': 're',
    'pure-re': 're',
#     'code-ner': 'ner',
#     'code-re': 're',
#     'code-pure-re': 're',
}

train_file = {
    'scierc': "/home/jc/workspace/exp/data/sciERC/processed_data/train.json",
    'nyt': "/home/jc/workspace/exp/data/nyt10/processed_data/nyt10_train.txt",
    'tacred': "/home/jc/workspace/exp/data/tacred/train.txt",
    'webnlg': "/home/jc/workspace/exp/data/webnlg/en/processed_data/train.json",
    'conll2003': '/home/jc/workspace/exp/data/conll2003/processed_data/train.json',
    'wikigold': '/home/jc/workspace/exp/data/wikigold/processed_data/test.json',
    # 'wikigold': 'Wiki/processed_data/train.json',
    'ontonotes': '/home/jc/workspace/exp/data/OntoNotes/processed_data/train.json',
    # 'ontonotes': 'OntoNote/processed_data/train.json',
    'bbn': '/home/jc/workspace/exp/data/BBN/processed_data/train.json',
    'ace2005ner': '/home/jc/workspace/exp/data/ace2005/ner/processed_data/train.json',
    'ace2005re': '/home/jc/workspace/exp/data/ace2005/re/processed_data/train.json',
    'conll2004': '/home/jc/workspace/exp/data/conll2004/processed_data/train.json',
}

class Prompter:
    def __init__(self, dataset, task, shot, kb_path=None, kb_thres=None, 
                 strategy='random', prompt_style='natural'):
        self.dataset = dataset
        self.task = task
        self.shot = shot
        self.prompt_strat = strategy
        self.dh = DataHandler(self.dataset, self.task,
                              data_file=train_file[self.dataset])
        self.dh.load_random_test_data()
        self.kb = KBHandler(task, shot, kb_thres, kb_path=kb_path) if kb_path else None

        self.prompt_style = prompt_style

    def get_prompt(self):
        if self.prompt_strat == 'random':
            return self._get_prompt_random()
        elif self.prompt_strat == 'type-aware': # sample_num = shot * type_num
            return self._get_prompt_type_aware()

    def _get_prompt_random(self):
        shots = []
        samples = self.dh.df.sample(self.shot, random_state=random.randint(100, 999), axis=0)
        if self.prompt_style == 'python':
            shots = self._build_python_shots(samples)
        elif self.prompt_style == 'natural':
            shots = self._build_natural_shots(samples)
        return "".join(shots) + template_base[self.task+"_"+self.prompt_style]
        # elif self.prompt_style == 'json':
        #     shots = self._build_json_shots(samples)
        #     return shots
    
    def _get_prompt_type_aware(self):
        shots = []
        types = self.get_type()
        for type_item in types:  # for each type
            # Filter samples with specific type
            def contains_substring(lst, substring):
                return any(substring in tup for tup in lst)
            if task_map[self.task] == 'ner' and type_item == 'NA':
                mask = self.dh.df[task_map['ner']].apply(lambda x: len(x) == 0)
            else:
                mask = self.dh.df[task_map[self.task]].apply(contains_substring, substring=type_item)   # apply filtering condition to 'ner' or 're' column
            samples = self.dh.df[mask].sample(self.shot, random_state=random.randint(100, 999), axis=0, replace=True) # replace=True to ensure enough samples
            
            if self.prompt_style == 'python':
                shots += self._build_python_shots(samples)
            elif self.prompt_style == 'natural':
                shots += self._build_natural_shots(samples)
        random.shuffle(shots)
        return "".join(shots) + template_base[self.task+"_"+self.prompt_style]
            # elif self.prompt_style == 'json':
            #     new_shots = self._build_json_shots(samples)
            #     # Todo

    def get_prompt_similar(self, input_sentence, label):
        """Get prompts that are structurally similar to the input sentence"""
        samples = self.kb.get_struc_sim(input_sentence, label)
        hits = len(samples)
        print("Hits: ", hits)

        if len(samples) < self.shot:
            supplement = self.dh.df.sample(self.shot-len(samples), random_state=random.randint(100, 999), axis=0)
            samples = pd.concat([supplement, samples])

        if self.prompt_style == 'python':
            shots = self._build_python_shots(samples)
        elif self.prompt_style == 'natural':
            shots = self._build_natural_shots(samples)
        return "".join(shots) + template_base[self.task+"_"+self.prompt_style], hits
        # elif self.prompt_style == 'json':
        #     shots = self._build_json_shots(samples)
        #     return shots, hits

    def _build_python_shots(self, samples):
        """Build python style shots from samples"""
        shots = []
        for i in range(self.shot):
            tmp_shot = template_base[self.task+"_"+self.prompt_style]
            sample = samples.iloc[i]
            sentence = sample['sentence']
            if self.task == 'pure-re':
                entity_list = [(head, tail) for head, _, tail in sample[task_map[self.task]]]
                tmp_shot += "\tentity_list={}\n".format(entity_list)
            
            labels = sample[task_map[self.task]] # label = sample['ner']
            for j in range(len(labels)): # len(sample['ner']
                if task_map[self.task] == 'ner':
                    tmp_shot += "\tentity_list.append({{{{\"text\": \"" + labels[j][0] + "\", \"type\": \"" + labels[j][1] + "\"}}}})\n"
                elif task_map[self.task] == 're':
                    # if self.task == 'code-pure-re':
                        # tmp_shot += "\tent1_text={}\n\tent2_text={}\n".format(label[j][0], label[j][2])
                    tmp_shot += "\tentity_relation_list.append({{{{\"rel_type\": \"" + labels[j][1] + "\", \"ent1_text\": \"" + labels[j][0] + "\", \"ent2_text\": \"" + labels[j][2] + "\"}}}})\n"
            # try:
            tmp_shot = tmp_shot.format_map({'sentence': sentence})
            # except Exception as e:
            #     print("Error when formatting shot: ", tmp_shot)
            #     print("Error: ", e)
            #     print("Sample: ", sample)
            shots.append(tmp_shot)
        return shots

    def _build_natural_shots(self, samples):
        """Build natural language style shots from samples"""
        shots = []
        for i in range(self.shot):
            tmp_shot = template_base[self.task+"_"+self.prompt_style]
            sample = samples.iloc[i]
            sentence = sample['sentence']
            if self.task == 'pure-re':
                entity_list = [(head, tail) for head, _, tail in sample[task_map[self.task]]]
                tmp_shot += "\tentity_list: {}\n".format(entity_list)
            
            labels = sample[task_map[self.task]]
            tmp_shot += "\tExtracted results: " + str(labels) + "\n"
            tmp_shot = tmp_shot.format_map({'sentence': sentence})
            shots.append(tmp_shot)
        return shots
    
    def _build_json_shots(self, samples):
        """
        Build json style shots from samples
        NER: {1: {
                "sentence": "I am a student.",
                "entity_list": [{"text": "student", "type": "PER"}, ...]
            }, 2: {}, ...}
        RE: {1: {
                "sentence": "I am a student.",
                "relation_list": [{"rel_type": "work_for", "ent1_text": "student", "ent2_text": "school"}, ...]
            }, 2: {}, ...}
        PURE-RE: {1: {
                "sentence": "I am a student.",
                "entity_list": [("student", "school"), ...],
                "relation_list": [{"rel_type": "work_for", "ent1_text": "student", "ent2_text": "school"}]
            }, 2: {}, ...}
        """
        shots = {}
        for i in range(self.shot):
            tmp_shot = {'sentence': ""}
            if task_map[self.task] == 'ner':
                tmp_shot['entity_list'] = []
            elif task_map[self.task] == 're':
                tmp_shot['relation_list'] = []
            sample = samples.iloc[i]
            sentence = sample['sentence']
            if self.task == 'code-pure-re':
                entity_list = [(head, tail) for head, _, tail in sample[task_map[self.task]]]
                tmp_shot['entity_list'] = entity_list

            labels = sample[task_map[self.task]]
            for j in range(len(labels)):
                if task_map[self.task] == 'ner':
                    tmp_shot['entity_list'].append({'text': labels[j][0], 'type': labels[j][1]})
                elif task_map[self.task] == 're':
                    tmp_shot['relation_list'].append({'rel_type': labels[j][1], 'ent1_text': labels[j][0], 'ent2_text': labels[j][2]})
            tmp_shot['sentence'] = sentence
            shots[str(i)] = tmp_shot
        return shots

    def get_type(self):
        max_shot_num = 25
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "type_dict_of_test.json"), 'r') as f:
            type_dict = json.load(f)
            f.close()
        types = type_dict[self.dataset][task_map[self.task]].keys()
        try:
            del types['NA']
        except:
            pass
        if len(types) > max_shot_num:
            sorted_types = [k for k, v in sorted(types.items(), key=lambda item: item[1], reverse=True)][:max_shot_num]
        else:
            sorted_types = list(types)
        return sorted_types

    def filter_type(self):
        """To filter types from every * test (not train) * sample"""
        dh = DataHandler(self.dataset, self.task)
        dh.load_random_test_data()
        if self.dataset in ['conll2003', 'wikigold', 'ontonotes', 'bbn', 'ace2005ner']:   # NER Datasets
            ner = list(dh.df['ner'])
            null_mask = dh.df['ner'].apply(lambda ner: len(ner) == 0)
            ner = [item for sublist in ner for item in sublist] # flatten
            
            type_count = defaultdict(int)
            for _, typee in ner:
                type_count[typee] += 1
            type_count['NA'] = len(dh.df[null_mask]) # no target as a single type
            return type_count

            # ner_type = set([item[1] for item in ner]) # get unique types
            # ner_type.add('NA')  # no target as a single type
            # return list(ner_type)
        elif self.dataset in ['nyt', 'tacred', 'ace2005re', 'conll2004']: # RE Datasets
            re = list(dh.df['re'])
            re = [item for sublist in re for item in sublist]
            
            type_count = defaultdict(int)
            for _, typee, _ in re:
                type_count[typee] += 1
            # type_count['NA'] = type_count['']
            # del type_count['']
            return type_count
        elif self.dataset in ['scierc']:    # NER & RE Datasets
            ner = list(dh.df['ner'])
            null_mask = dh.df['ner'].apply(lambda ner: len(ner) == 0)
            ner = [item for sublist in ner for item in sublist] # flatten

            re = list(dh.df['re'])
            re = [item for sublist in re for item in sublist]
            type_count_ner = defaultdict(int)
            type_count_re = defaultdict(int)
            for _, typee in ner:
                type_count_ner[typee] += 1
            for _, typee, _ in re:
                type_count_re[typee] += 1
            return type_count_ner, type_count_re
        else:
            raise NotImplementedError("Dataset {} is not implemented to filter types.".format(self.dataset))

def parse_type(datasets):
    """Parse types of entity/ relation in test set and save to json file."""
    type_dict = {}
    for dataset in datasets:
        p = Prompter(dataset, 're', 2, '', '')
        types = p.filter_type()
        type_dict[dataset] = {}
        if dataset in ['scierc']:
            type_dict[dataset]['ner'] = types[0]
            type_dict[dataset]['re'] = types[1]
        elif dataset in ['conll2003', 'wikigold', 'ontonotes', 'bbn', 'ace2005ner']:
            type_dict[dataset]['ner'] = types
        elif dataset in ['nyt', 'tacred', 'ace2005re', 'conll2004']:
            type_dict[dataset]['re'] = types

    script_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_path, "type_dict_of_test.json"), 'w') as f:
        json.dump(type_dict, f)
        f.close()

if __name__ == '__main__':
    # p = Prompter('conll2003', 'ner', 2, strategy='type-aware', prompt_style='natural')
    # string = "I am a student."
    # print(p.get_prompt())

    # datasets = ['conll2003', 'ontonotes', 'bbn', 
    #             'scierc', 'nyt', 'tacred', 'conll2004',]
    # parse_type(datasets)

    ### update kb
    datasets = {}
    datasets['ner'] = ['conll2003', 'ontonotes', 'scierc', 'ace2005ner']
    datasets['re'] = ['nyt', 'tacred', 'scierc', 'ace2005re', 'conll2004']
    tasks = ['ner', 're']
    shot = 500
    for task in tasks:
        for dataset in datasets[task]:
            kb_path = f'/home/jc/workspace/exp/kbase/sim_{task}_{dataset}.csv'
            kb = KBHandler(task, shots=1, kb_path=kb_path)

            dh = DataHandler(dataset, task, line_num=shot, data_file=train_file[dataset])
            dh.load_random_test_data()
            sentences, labels = dh.df['sentence'].tolist(), dh.df[task_map[task]].tolist()
            kb.update_struc_kb(sentences, labels)
    

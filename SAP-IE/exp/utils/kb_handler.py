# jc 23.12.03

import json
import os
import random

import spacy
from spacy import displacy
import zss
import time
import pandas as pd

# 加载 spaCy 的英文模型
# nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])
nlp = spacy.load('en_core_web_sm')#, disable=['ner'])

task_map = {
    'ner': 'ner',
    're': 're',
    'pure-re': 're',
}

class KBHandler:
    '''Knowledge Base Handler'''
    def __init__(self, task, shots, thres=10, mode='sim', score_kb_max_line=500,
                 kb_path='/home/jc/workspace/exp/kbase/task_template.csv',):
        print("kb_path: ", kb_path)
        self.kb_path = kb_path
        self.kb = None
        self.score_kb_max_line = score_kb_max_line
        self.mode = mode
        self._load_kb()

        self.thres = thres
        self.task = task_map[task]
        self.shots = shots

    def _load_kb(self):
        if not os.path.exists(self.kb_path):    # create file if not exists
            open(self.kb_path, 'a').close()
        with open(self.kb_path, 'r') as f:
            try:
                self.kb = pd.read_csv(f, sep='\t')
                self.kb['root'] = self.kb['sentence'].apply(lambda x: get_tree(x))  # calculate roots for every sentence
            except Exception as e:
                print(">>> Error: ", e)
                if self.mode == 'sim':
                    self.kb = pd.DataFrame(columns=['sentence', 'task', 'label', 'root'])
                elif self.mode == 'check':
                    # ###self.kb = pd.DataFrame(columns=['sentence', 'task', 'pred', 'score', 'root'])
                    self.kb = pd.DataFrame(columns=['sentence', 'task', 'pred', 'label', 'root'])
            f.close()
        
    def _save_kb(self):
        kb_to_save = self.kb.drop('root', axis=1)   # remove root column
        try:
            kb_to_save = kb_to_save.drop('similarity', axis=1)
        except:
            pass
        kb_to_save.to_csv(self.kb_path, sep='\t', index=False)

    def update_struc_kb(self, input_sentence, label):
        '''Update the structurally similar sentences to the KB'''
        for i in range(len(input_sentence)):
            new_row = pd.Series({'sentence': input_sentence[i], 'task': self.task, 'label': str(label[i])})
            self.kb = pd.concat([self.kb, pd.DataFrame(new_row).T])
            self._save_kb()
        print(">>> KB updated. ")

    def get_struc_sim(self, input_sentence, label, insert2kb=False):
        '''Get the structurally similar sentences from the KB'''
        input_root = get_tree(input_sentence)
        self.kb['similarity'] = self.kb['root'].apply(lambda x: calculate_similarity_fast(x, input_root))
        mask = (self.kb['similarity'] <= self.thres) # filter similarity
        # mask = self.kb['sentence'].apply(lambda x: calculate_similarity(x, input_sentence) <= self.thres) # filter similarity
        # mask_task = self.kb['task'].apply(lambda x: x == self.task)  # filter task
        mask_sent = self.kb['sentence'].apply(lambda x: x != input_sentence)  # filter input sentence
        mask = mask &  mask_sent# & mask_task
        samples = self.kb[mask]

        template_num = len(samples)
        shots = self.shots
        if template_num < shots:    # if not enough similar sentences, use all, and supplement with random sentences in prompts.py
            shots = template_num
            # print("+++ Not enough similar sentences in the KB. ")
            if insert2kb:
                new_row = pd.Series({'sentence': input_sentence, 'task': self.task, 'label': str(label)})
                self.kb = pd.concat([self.kb, pd.DataFrame(new_row).T])
                self.kb.loc[self.kb.index[-1], 'root'] = input_root
                self._save_kb()
        else:
            # print("+++ Found similar sentences in the KB. ")
            pass
        samples = samples.sample(shots, random_state=random.randint(1, 50), axis=0)[['sentence', 'label']]
        samples['label'] = samples['label'].apply(eval)     # eval lists in "label" column
        samples = samples.rename(columns={'label': self.task})   # rename "label" column to task name
        return samples

    def get_avg_sim(self):
        '''Get the average similarity of the kb'''
        num = 10
        samples = self.kb.sample(num, random_state=random.randint(1, 50), axis=0)
        sims = []
        for i in range(num):
            sample = samples.iloc[i]
            self.kb['similarity'] = self.kb['root'].apply(lambda x: calculate_similarity_fast(x, sample['root']))
            sim = self.kb['similarity'].mean()
            sims.append(sim)
        return sum(sims) / num

    # Initialize the KB with the first few sentences
    def update_score_kb(self, input_sentences, preds, scores):
        if self.score_kb_full() <= 0:
            return
        for i in range(len(input_sentences)):
            input_sentence = input_sentences[i]
            input_root = get_tree(input_sentence)
            pred = preds[i]
            score = scores[i]
            new_row = pd.Series({'sentence': input_sentence, 'task': self.task, 'pred': str(pred), 'score': str(score)})
            self.kb = pd.concat([self.kb, pd.DataFrame(new_row).T])
            self.kb.loc[self.kb.index[-1], 'root'] = input_root
            self.kb = self.kb.drop_duplicates(subset=['sentence'], keep='last')
            self._save_kb()
        print(">>> Score KB updated. ")
    
    def update_revise_kb(self, input_sentences, preds, labels):
        if self.score_kb_full() <= 0:
            return
        for i in range(len(input_sentences)):
            input_sentence = input_sentences[i]
            input_root = get_tree(input_sentence)
            pred = preds[i]
            label = labels[i]
            new_row = pd.Series({'sentence': input_sentence, 'task': self.task, 'pred': str(pred), 'label': str(label)})
            self.kb = pd.concat([self.kb, pd.DataFrame(new_row).T])
            self.kb.loc[self.kb.index[-1], 'root'] = input_root
            self.kb = self.kb.drop_duplicates(subset=['sentence'], keep='last')
            self._save_kb()
        print(">>> Revise KB updated. ")

    def score_kb_full(self):
        """Return the number of lines left to be scored. <= 0 means full."""
        return self.score_kb_max_line - self.kb.shape[0]

    def get_answer_score(self):#, input_sentence):
        samples = self.kb

        template_num = len(samples)
        shots_to_retieve = self.shots
        if template_num < self.shots:
            shots_to_retieve = template_num
            # raise ValueError("Not enough similar sentences in the KB. ")
        # ###samples = samples.sample(shots_to_retieve, random_state=random.randint(1, 50), axis=0)[['sentence', 'pred', 'score']]
        samples = samples.sample(shots_to_retieve, random_state=random.randint(1, 50), axis=0)[['sentence', 'pred', 'label']]
        
        # if the last three ones are 0, swap them with the first three non-zero ones
        # while samples.iloc[-1]['score'] == 0 or samples.iloc[-2]['score'] == 0:
        #     samples = samples.sample(frac=1).reset_index(drop=True)
        # for i in range(3):
        #     if samples.iloc[-(i+1)]['score'] == 0:
        #         for j in range(len(samples) - 1):
        #             if samples.iloc[j]['score'] != 0:
        #                 samples.iloc[-(i+1)], samples.iloc[j] = samples.iloc[j].copy(), samples.iloc[-(i+1)].copy()
        #                 break
        return samples

def get_tree(sentence):
    # 使用 spaCy 解析句子，禁用不需要的步骤
    doc = nlp(sentence)
    # 获取句子的根词
    root = list(doc.sents)[0].root
    # 将 spaCy 的 Token 对象转换为 zss 的 Node 对象
    return convert_to_zss_node(root)

def convert_to_zss_node(token):
    # 创建一个新的 Node 对象
    # node = zss.Node(token.dep_)
    node = zss.Node(f"{token.text} ({token.dep_})")
    # 添加所有的子节点
    for child in token.children:
        node.addkid(convert_to_zss_node(child))
    return node

def calculate_similarity(sentence1, sentence2):
    """
    return: An integer distance [0, inf+)
    the smaller the distance, the more similar the sentences are
    """
    # 获取依赖树的根词
    root1 = get_tree(sentence1)
    root2 = get_tree(sentence2)
    # print("root1: ", root1)
    # print("root2: ", root2)
    # 计算结构相似度
    similarity = zss.simple_distance(root1, root2)
    # print(similarity)
    return similarity

def calculate_similarity_fast(root1, root2):
    return zss.simple_distance(root1, root2)

if __name__ == '__main__':
    ### print tree
    tree = get_tree("It is the time for Coste to return .")
    tree = get_tree("This is an example .")
    from anytree import Node, RenderTree
    def convert_to_anytree_node(zss_node):
        anytree_node = Node(zss_node.label)
        for zss_child in zss_node.children:
            anytree_child = convert_to_anytree_node(zss_child)
            anytree_child.parent = anytree_node
        return anytree_node
    anytree_root = convert_to_anytree_node(tree)
    for pre, fill, node in RenderTree(anytree_root):
        print("%s%s" % (pre, node.name))

    ### cal similarity
    # stan = "Rosy is a student from China ."
    # sentences = ["Jerry is a teacher from America .",
    #              "Tom is a doctor from England .",
    #              "Rabbin is a lawyer from India .",
    #              "Miss White studied in Oxford University .",
    #              "The first time I met him was in the library ."]
    # for i in range(len(sentences)):
    #     print(calculate_similarity(stan, sentences[i]))

    ### retrieve from KB
    # sentence1 = "In the interim , Toronto and Montreal built Canada 's dynasty franchises , and Edmonton created an era of its own in the '80s , the Wayne Gretzky years ."
    # sentence2 = "In Colorado three theater companies -- the Curious Theater Company and Paragon Theater , both in Denver , and Theater13 in Boulder -- have gone so far as to sue the state , arguing that smoking in the course of a play is a form of free expression ."
    # time0 = time.time()
    # kb = KBHandler('ner', 25, 15, kb_path='/home/jc/workspace/exp/kbase/sim_ner_ontonotes.csv')
    # time1 = time.time()
    # print("Loading time: ", time1 - time0)

    # samples = kb.get_struc_sim(sentence1, None, False)
    # time2 = time.time()
    # print("Running time: ", time2 - time1)
    # print("Samples: ", samples)
        

    ### cal avg sim
    # datasets = {}
    # datasets['ner'] = ['conll2003', 'ontonotes', 'scierc', 'ace2005ner']
    # datasets['re'] = ['nyt', 'tacred', 'scierc', 'ace2005re']
    # tasks = ['ner', 're']
    # avg_sims = {}
    # for task in tasks:
    #     for dataset in datasets[task]:
    #         kb = KBHandler(task, 25, 15, kb_path=f'/home/jc/workspace/exp/kbase/sim_{task}_{dataset}.csv')
    #         avg_sim = kb.get_avg_sim()
    #         avg_sims[str((task, dataset))] = avg_sim
    #         print(f"task: {task}, dataset: {dataset}")
    #         print(avg_sim)
    #         print("===")
    # with open('/home/jc/workspace/exp/prompt/avg_sims.json', 'w') as f:
    #     json.dump(avg_sims, f)

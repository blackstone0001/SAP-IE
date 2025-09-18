# jc 23.11.09

import json
from datetime import datetime
import time
import sys
import os
import traceback
import spacy

from exp.model.model_interface import ModelInterface
from exp.utils.data_handler import *
from exp.test.evaluator import *
from exp.prompt.prompter import *
from exp.test.prompts import *
from exp.test.checker import *

task_map = {
    'ner': 'ner',
    're': 're',
    'pure-re': 're',
}

class TestRunner(ModelInterface):

    def __init__(self) -> None:
        super().__init__()
        self.test_line_num = self.args.test_line_num   # 在测试集中取的行数
        self.shot = self.args.shot
        self.dh = None  # DataHandler
        self.metric_type = None
        self.metric = None

        self.kb_time = 0
        self.kb_hit = 0

        self.args.check_rate = 0

    def run_test(self, dh: DataHandler):
        '''运行LLM (optional), 评估结果'''
        self.dh = dh
        try:
            if (not self.args.check_fact) or (self.args.check_fact and len(self.args.out_file_to_check_path) == 0):
                self.get_test_output()
                self.eval_test()
                # if self.args.check_fact:
                #     self.check_fact(self.target, self.prediction)
                self.record_result()

            elif self.args.check_fact and len(self.args.out_file_to_check_path) != 0:
                # self.metric, self.target, self.prediction = self.dh.load_output_file(self.args.out_file_to_check_path, comlumns=['sentence', task_map[self.args.task]])
                self.check_fact()
                filename = os.path.basename(self.args.out_file_to_check_path)[:-4] + f".checked-{self.shot}-shot.log"
                self.record_result(os.path.join(self.args.checked_out_dir, filename))
        except Exception as e:
            traceback.print_exc()
            print(">>> Eval or record Error: ", e)

    def get_test_output(self):
        self.dh.load_random_test_data()
        sentences = self.dh.df['sentence'].tolist()
        labels = self.dh.df[task_map[self.dh.task]].tolist()
        
        if self.args.kb:
            llm_input = self.gen_test_inputs_similar(sentences, labels)
        else:
            llm_input = self.gen_test_inputs(sentences, labels)
        self.run_batch(llm_input, self.dh.file_out, self.args.with_prompt)

    def gen_test_inputs(self, sentences, labels):
        '''
        Generate the inputs to LLM for test
        Same prompt for all sentences
        '''
        template = Prompter(self.args.dataset, self.args.task, 
                            self.args.shot, strategy=self.args.prompt_strat, 
                            prompt_style=self.args.prompt_style).get_prompt()
        inputs = []
        for index, sentence in enumerate(sentences):
            input_text = template.format_map({'sentence': sentence})
            if self.args.task == 'pure-re':    # Give the entity pair
                entity_list = [(head, tail) for head, _, tail in labels[index]]
                if self.args.prompt_style == 'python':
                    input_text += "\tentity_list={}\n".format(entity_list)
                elif self.args.prompt_style == 'natural':
                    input_text += "\tentity_list: {}\n".format(entity_list)
            # elif self.args.prompt_style == 'json':
            #     input_text = {
            #         'task': self.args.task,
            #         'given_type_list': [],
            #         'result': template
            #     }
            #     input_text['result'][str(self.shot)] = {'sentence': sentence}
            #     if self.args.task == 'code-pure-re':
            #         input_text['result'][str(self.shot)]['entity_list'] = [(head, tail) for head, _, tail in labels[index]]
            #     input_text = json.dumps(input_text)[:-3]
            #     print(">>> input_text: ", input_text)

            inputs.append(input_text)
        return inputs
    
    def gen_test_inputs_similar(self, sentences, labels):
        '''
        Generate shots that are structurally similar to the input sentence
        '''
        inputs = []
        total_time, total_hit = 0, 0
        p = Prompter(self.args.dataset, self.args.task, 
                     self.args.shot, self.args.kb_path,
                     self.args.kb_thres, self.args.prompt_strat, 
                     prompt_style=self.args.prompt_style)
        for index, sentence in enumerate(sentences):
            tic = time.time()
            template, hits = p.get_prompt_similar(sentence, labels[index])
            tok = time.time()
            total_time += tok - tic
            total_hit += hits

            input_text = template.format_map({'sentence': sentence})
            if self.args.task == 'pure-re':    # Give the entity pair
                entity_list = [(head, tail) for head, _, tail in labels[index]]
                if self.args.prompt_style == 'python':
                    input_text += "\tentity_list={}\n".format(entity_list)
                elif self.args.prompt_style == 'natural':
                    input_text += "\tentity_list: {}\n".format(entity_list)
                # re = labels[index][0]
                # input_text += "\tent1_text={}\n\tent2_text={}\n".format(re[0], re[2])
            inputs.append(input_text)
        self.kb_time = round(total_time/len(sentences), 2)
        self.kb_hit = round(total_hit/(len(sentences)*self.shot), 2)
        print(">>> Average KB retrieval time: ", round(total_time/len(sentences), 2))
        print(">>> Average KB retrieval hits: ", round(total_hit/(len(sentences)*self.shot), 2))
        return inputs

    def eval_test(self):
        '''评估LLM的输出'''
        target, sentences, pred = self.dh.prepare_eval_data(self.args.prompt_style, self.args.use_api)
        self.target = target    # Dataframe, including sentence and label
        self.prediction = pred  # Including only predicted label

        evaluator = Evaluator(list(target[task_map[self.args.task]]), pred, self.args.metric_type)
        metric = evaluator.evaluate(self.dh.dataset, self.dh.task)
        print(">>> {}: ".format(self.args.metric_type), metric, "%")
        self.metric_type, self.metric = evaluator.metric_type, metric

    def check_fact(self):
        self.ck = Checker(self.dh.dataset, self.dh.task, self.shot, 
                          self.args.out_file_to_check_path, self.args.kb_path)
        self.ck.annotate()  # Load original prediction and target, and update the KB
        self.target, self.prediction = self.ck.target, self.ck.pred # For recording
        sentence = self.ck.target['sentence']
        inputs = []
        promtp_temp = self.ck.get_check_prompt()
        for i in range(len(sentence)):  # Build checking prompt for each sentence
            # print(f"{i}-th line: ", ck.get_check_prompt(sentence[i]))
            # input_str = self.ck.get_check_prompt(sentence[i]).format_map({'sentence': sentence[i], 'answer': str(self.ck.pred[i])})
            input_str = promtp_temp.format_map({'sentence': sentence[i], 'answer': str(self.ck.pred[i])})
            inputs.append(input_str)
        try:
            checked = self.run_batch(inputs, None, with_prompt=False)
            self.checked = self.dh.prepare_check_fact_data(checked, self.args.prompt_style, 
                                                           sentence_head="<The given sentence>:",
                                                           shot_head="<The given sentence>:",
                                                           answer_head="Reviewed answer",
                                                           match_str=r'\[.*?\]')
            print(">>> Checked: ", self.checked)
        except Exception as e:
            traceback.print_exc()

        # checked_num = self.post_process()     # Review checked answers
        checked_num = self.post_revise()     # Review checked answers
        self.args.check_rate = round(checked_num/len(self.checked), 3)

        e = Evaluator(list(self.ck.target[task_map[self.args.task]]), self.ck.pred, self.args.metric_type)
        print(">>> Unchecked: ", e.evaluate(self.dh.dataset, self.dh.task))

        evaluator = Evaluator(list(self.ck.target[task_map[self.args.task]]), self.checked, self.args.metric_type)#self.checked, self.args.metric_type)
        metric = evaluator.evaluate(self.dh.dataset, self.dh.task)
        print(">>> Checked: {}: ".format(self.args.metric_type), metric, "%")

        self.metric = self.ck.metric
        self.metric_type, self.check_metric = evaluator.metric_type, metric

    def post_process(self):
        checked_num = 0
        for i in range(len(self.checked)):
            if self.dh.dataset == 'nyt' or self.dh.dataset == 'tacred':    ### NOT abandon for NYT/ tacred
                if self.checked[i] != 10:
                    if self.checked[i] > 5:
                        self.checked[i] = get_sim_type(self.ck.pred[i], Type_Options[self.dh.dataset][task_map[self.dh.task]])
                    else:
                        self.checked[i] = get_sim_type(self.ck.pred[i], Type_Options[self.dh.dataset][task_map[self.dh.task]], 9)
                else:
                    self.checked[i] = self.ck.pred[i]
                
            elif self.checked[i] == 0:
                self.checked[i] = []
            else:
                self.checked[i] = self.ck.pred[i]
            # for index, entity in enumerate(self.checked[i]):    # Remove the entity that is not in the target or not in the type list
            #     if entity[0] not in target.iloc[i,0]:
            #         self.checked[i].pop(index)
            #     if entity[1] not in type_list:
            #         self.checked[i].pop(index)
            if self.checked[i] != self.ck.pred[i]:
                checked_num += 1
        return checked_num

    def post_revise(self):
        checked_num = 0
        for i in range(len(self.checked)):
            if self.checked[i] != self.ck.pred[i]:
                checked_num += 1
        return checked_num

    def record_result(self, custom_file_name=None):
        content, contrast = [], []
        content.append("Time: " + str(datetime.now()))
        content.append("Dataset: " + self.dh.dataset)
        content.append("Prompt Style: " + self.args.prompt_style)
        content.append("Task: " + self.dh.task)
        content.append("Model: " + self.args.model_name)
        content.append("Shot: " + str(self.shot))
        content.append("Generation Config: " + str(self.generation_config))
        content.append("Metric({}): ".format(self.metric_type) + str(self.metric) + "%")
        if self.args.check_fact:
            content.append("Checked Metric({}): ".format(self.metric_type) + str(self.check_metric) + "%")

        if self.args.kb:
            content.append("KB average retrieval time: " + str(self.kb_time))
            content.append("KB average retrieval hits: " + str(self.kb_hit))
            content.append("KB similarity threshold: " + str(self.args.kb_thres))
        
        if self.args.check_fact:
            content.append("Check rate: " + str(self.args.check_rate))
            # content.append("Check accuracy: " + str(self.args.check_acc))

        content.append("\n\n\n")
        content.append("target, prediction, (checked)")
        for i in range(len(self.target)):
            string = self.target.iloc[i,0]+"\n"+str(self.target.iloc[i][task_map[self.args.task]])+"\n"+str(self.prediction[i])+"\n"
            if self.args.check_fact:
                string += str(self.checked[i])+"\n"
            content.append(string)

        if custom_file_name:
            file_out = custom_file_name
        else:
            file_out = self.dh.file_out+".log"
        self.dh.write_data_file(file_out, content, contrast)

def get_sim_type(pred, type_list, rand_num=1):
    """Get similar type but not the same type, for NYT and TACRED dataset"""
    original_type = pred[0][1]
    def semantic_sim(s1, s2):
        # s1, s2 = s1.split(':')[1], s2.split(':')[1]
        doc1 = nlp(s1)
        doc2 = nlp(s2)
        return doc1.similarity(doc2)

    sim_dict = {}
    for t in type_list:
        if t == original_type:
            continue
        sim = semantic_sim(original_type, t)
        sim_dict[t] = sim
    sim_dict = dict(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True))
    print(sim_dict)
    ind = random.randint(0, rand_num)
    sim_type = list(sim_dict.keys())[ind]
    return [(pred[0][0], sim_type, pred[0][2])]

# def get_random_type(pred, type_list):
#     """Get a random type from the type list"""
#     return [(pred[0][0], random.choice(type_list), pred[0][2])]
    
if __name__ == '__main__':
    # t = TestRunner()
    # pred = [('Fidel Castro', '/location/neighborhood/neighborhood_of', 'Cuba')]
    # types = Type_Options['nyt']['re']
    # # types = Type_Options['tacred']
    # print(get_sim_type(pred, types))
    # print(get_sim_type(pred, types, 9))


    # to manually check the output and record
    checked = []

    origin_file_path = "/home/jc/workspace/exp/figs/data/llama/ner/scierc/base/scierc_ner_natural_llama-7b_shot-15_maxline-300_rpenal-1.0_maxtoken-300_beam-1.json.1.log"
    for i in range(len(checked)):
        if not isinstance(checked[i], list):
            checked[i] = []
    sents, targs, unchecked, _, metrics_before = load_checked(origin_file_path, 'unchecked')
    p, r, f = eval_f1_micro(checked, targs)
    print(str((p, r, f)))

    task = 'ner'
    dataset = 'scierc'
    model = 'llama'
    # Manually write revised log
    tr = TestRunner()
    tr.dh = DataHandler(dataset=dataset, task=task)
    # tr.dh.set_output_path('/home/jc/workspace/exp/figs/data/llama/ner/scierc/revise/')
    tr.args.task = tr.dh.task
    tr.args.prompt_style = 'natural'
    tr.args.model_name = model
    tr.shot = 15
    tr.generation_config = ''
    tr.metric_type = 'f1'
    tr.metric = metrics_before
    tr.args.check_fact = True
    tr.check_metric = (p, r, f)
    tr.args.kb = False
    tr.kb_time = 0
    tr.kb_hit = 0
    tr.args.kb_thres = 0
    def count_check_rate(un, ch):
        checked_num = 0
        for i in range(len(un)):
            if un[i] != ch[i]:
                checked_num += 1
        return round(checked_num / len(un), 3)
    tr.args.check_rate = count_check_rate(unchecked, checked)
    tr.target = pd.DataFrame({'sentence': sents, task_map[tr.dh.task]: targs})
    tr.prediction = unchecked
    tr.checked = checked

    check_shot = 20
    filename = os.path.basename(origin_file_path)[:-4] + f".checked-{check_shot}-shot.log"
    out_path = f'/home/jc/workspace/exp/figs/data/{model}/{task}/{dataset}/revise/'
    custom_file_name = os.path.join(out_path, filename)
    # print(custom_file_name)
    tr.record_result(custom_file_name=custom_file_name)

# def levenshtein_distance(s1, s2):
#     if len(s1) < len(s2):
#         return levenshtein_distance(s2, s1)
#     if len(s2) == 0:
#         return len(s1)
#     previous_row = range(len(s2) + 1)
#     for i, c1 in enumerate(s1):
#         current_row = [i + 1]
#         for j, c2 in enumerate(s2):
#             insertions = previous_row[j + 1] + 1
#             deletions = current_row[j] + 1
#             substitutions = previous_row[j] + (c1 != c2)
#             current_row.append(min(insertions, deletions, substitutions))
#         previous_row = current_row
#     return previous_row[-1]

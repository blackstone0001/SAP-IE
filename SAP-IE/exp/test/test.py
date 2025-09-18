# jc 2023.10.30 15:16

import sys
sys.path.append('/home/jc/workspace/')

from exp.utils.settings import *
from exp.test.test_runner import *

def test():
    tr = TestRunner()
    dh = DataHandler(dataset=tr.args.dataset, 
                     task=tr.args.task,
                     line_num=tr.args.test_line_num,
                     data_file=tr.args.data_file,
                     FILE_OUT_DIR=tr.args.out_dir)

    file_prefix = "{}_{}_{}_{}_shot-{}_maxline-{}_rpenal-{}_maxtoken-{}_beam-{}".format(
                    dh.dataset, dh.task, tr.args.prompt_style, tr.args.model_name, tr.shot, 
                    tr.test_line_num, tr.generation_config.repetition_penalty, 
                    tr.generation_config.max_new_tokens, tr.generation_config.num_beams)
    if tr.args.do_sample:
        file_prefix += "_temp-{}_k-{}_p-{}".format(
                        tr.generation_config.temperature, tr.generation_config.top_k,
                        tr.generation_config.top_p)
        
    file_out = os.path.join(dh.FILE_OUT_DIR, file_prefix + ".json" + tr.args.out_file_suffix)
    dh.set_output_path(file_out)
    tr.run_test(dh)

if __name__ == '__main__':
    test()


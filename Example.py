#!/usr/bin/env python
# coding: utf-8

import model.config as config
from gerbil.nn_processing import NNProcessing

class arguments:
    def __init__(self):
        self.all_spans_training = True
        self.build_entity_universe = False
        self.el_mode = True
        self.el_with_stanfordner_and_our_ed = False
        self.entity_extension = 'extension_entities'
        self.experiment_folder = config.base_folder + '/data/tfrecords/paper_models/'
        self.experiment_name = 'paper_models'
        self.hardcoded_thr = None
        self.lowercase_spans_pem = False
        self.output_folder = config.base_folder + '/data/tfrecords/paper_models/all_spans_training_folder/base_att_global/'
        self.persons_coreference = True
        self.persons_coreference_merge = True
        self.running_mode = None
        self.training_name = 'base_att_global '


class train_arguments:
    def __init__(self):
        self.all_spans_training = True
        self.attention_K = 100
        self.attention_R = 10
        self.attention_ent_vecs_no_regularization = True
        self.attention_on_lstm = False
        self.attention_retricted_num_of_entities = None
        self.attention_use_AB = False
        self.batch_size = 3
        self.cand_ent_num_restriction = None
        self.checkpoints_folder = config.base_folder + '/data/tfrecords/paper_models/all_spans_training_folder/base_att_global/checkpoints/'
        self.checkpoints_num = 1
        self.clip = -1
        self.comment = ''
        self.continue_training = False
        self.debug = False
        self.dim_char = 50
        self.dropout = 0.5
        self.ed_datasets = None
        self.ed_val_datasets = [1]
        self.el_datasets = ['aida_train.txt', 'aida_dev.txt', 'aida_test.txt', 'ace2004.txt', 'aquaint.txt',
                            'msnbc.txt'],
        self.el_val_datasets = [1]
        self.ent_vecs_regularization = 'l2dropout'
        self.entity_extension = 'extension_entities'
        self.eval_cnt = 39
        self.evaluation_minutes = 10
        self.experiment_name = 'paper_models'
        self.fast_evaluation = True
        self.feature_size = 20
        self.ffnn_dropout = True
        self.ffnn_l2maxnorm = None
        self.ffnn_l2maxnorm_onlyhiddenlayers = False
        self.final_score_ffnn = [0, 0]
        self.gamma_thr = 0.2
        self.global_gmask_based_on_localscore = False
        self.global_gmask_unambigious = False
        self.global_mask_scale_each_mention_voters_to_one = False
        self.global_norm_or_mean = 'norm'
        self.global_one_loss = False
        self.global_score_ffnn = [0, 0]
        self.global_thr = 0.0
        self.global_topk = None
        self.global_topkfromallspans = None
        self.global_topkthr = None
        self.hardcoded_thr = None
        self.hidden_size_char = 50
        self.hidden_size_lstm = 150
        self.improvement_threshold = 0.3
        self.inconsistent_model_folder = True
        self.lr = 0.001
        self.lr_decay = -1.0
        self.lr_method = 'adam'
        self.max_mention_width = 10
        self.model_heads_from_bilstm = False
        self.nepoch_no_imprv = 6
        self.nn_components = 'pem_lstm_attention_global'
        self.no_p_e_m_usage = False
        self.nocheckpoints = False
        self.onleohnard = False
        self.output_folder = config.base_folder + '/data/tfrecords/paper_models/all_spans_training_folder/base_att_global/'
        self.pem_buckets_boundaries = None
        self.pem_without_log = False
        self.running_mode = 'gerbil'
        self.shuffle_capacity = 500
        self.span_boundaries_from_wordemb = False
        self.span_emb = 'boundaries'
        self.span_emb_ffnn = [0, 0]
        self.stage2_nn_components = 'local_global'
        self.steps_before_evaluation = 10000
        self.summaries_folder = config.base_folder + '/data/tfrecords/paper_models/all_spans_training_folder/base_att_global/summaries/'
        self.train_datasets = ['aida_train.txt']
        self.train_ent_vecs = False
        self.training_name = 'group_global/c50h50_lstm150_nohead_attR10K100_fffnn0_0_glthr00_glffnn0_0v1'
        self.use_chars = True
        self.use_features = False
        self.zero = 1e-06


args = arguments()
train_args = train_arguments()

nnprocessing = NNProcessing(train_args, args)

sentence = '''Brexit Party founder Catherine Blaiklock who resigned from the party after posting a series of anti-Islam comments has backed Boris Johnson’s Conservatives.

Ms Blaiklock, who set up the party and registered its name, accused Nigel Farage of going on a “monumental ego trip” and said his general election strategy had been a “disaster”.

“Nigel has failed catastrophically,” she told The Sun newspaper. “You have to compromise. If you want Brexit, you must vote Tory."

But opposition parties at Westminster seized on the endorsement, and the Liberal Democrat deputy leader Ed Davey said: “Catherine joins a long list of unsavoury characters, including Tommy Robinson, who are now backing Boris Johnson.”

He added: “The fact that Sir John Major, Michael Heseltine and others are urging voters to keep the Tories out of power shows quite how far this Conservative party has sunk.”

Ms Blaiklock resigned from the Brexit Party earlier this year after a series of anti-Islam messages were uncovered by the Hope Not Hate organisation, which monitors the the far-right.

According to The Guardian, one of the messages shared by Ms Blaiklock was from a former BNP acivtist which referred to “white genocide” while one of her own remarks read: “Islam = submission – mostly raping men it seems”.'''

print('processing:')
print(sentence)

out = nnprocessing.process(sentence, [])

print('results:')
print(out)

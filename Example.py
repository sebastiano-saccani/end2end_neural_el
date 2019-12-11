#!/usr/bin/env python
# coding: utf-8

import model.config as config
from gerbil.nn_processing import NNProcessing
from model.util import load_train_args
import argparse

def _parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="per_document_no_wikidump",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", default="doc_fixed_nowiki_evecsl2dropout")
    parser.add_argument("--all_spans_training", type=bool, default=False)
    parser.add_argument("--el_mode", dest='el_mode', action='store_true')
    parser.add_argument("--ed_mode", dest='el_mode', action='store_false')
    parser.set_defaults(el_mode=True)

    parser.add_argument("--running_mode", default=None, help="el_mode or ed_mode, so"
                                                             "we can restore an ed_mode model and run it for el")

    parser.add_argument("--lowercase_spans_pem", type=bool, default=False)

    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    # those are for building the entity set
    parser.add_argument("--build_entity_universe", type=bool, default=False)
    parser.add_argument("--hardcoded_thr", type=float, default=None, help="0, 0.2")
    parser.add_argument("--el_with_stanfordner_and_our_ed", type=bool, default=False)

    parser.add_argument("--persons_coreference", type=bool, default=False)
    parser.add_argument("--persons_coreference_merge", type=bool, default=False)
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if args.persons_coreference_merge:
        args.persons_coreference = True
    # print(args)
    if args.build_entity_universe:
        return args, None

    temp = "all_spans_" if args.all_spans_training else ""
    args.experiment_folder = config.base_folder + "data/tfrecords/" + args.experiment_name + "/"
    print('base folder ######',config.base_folder)
    args.output_folder = (config.base_folder + "data/tfrecords/" + args.experiment_name + "/{}training_folder/".format(
        temp) + args.training_name + "/").replace(" ", "")

    train_args = load_train_args(args.output_folder, "gerbil")
    train_args.entity_extension = args.entity_extension

    # print(train_args)
    return args, train_args


args, train_args = _parse_args()

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
print('result')
print(out)

import  OpenKE.config as config
import OpenKE.models as models
import tensorflow as tf
import numpy as np
import json
import os
import sys
import time


def test(strategy):
    data_set = ['dbpe', 'lex', 'yago']
    for dataset in data_set:
        for round_num in range(1, 8):
            # (1) Set import files and OpenKE will automatically load models via tf.Saver().
            con = config.Config()
            con.set_in_path('./OpenKE/benchmarks/' + dataset + '_' + strategy + '/')
            con.set_test_link_prediction(True)
            con.set_work_threads(8)
            con.set_dimension(100)
            con.set_import_files('./OpenKE/benchmarks/' + dataset + '_' + strategy + '/model/' + str(round_num) + '/model.vec.tf')
            con.init()
            con.set_model(models.TransE)
            def get_test_entity(count_nums):
                entity_set = set()
                count = 0
                with open('./OpenKE/benchmarks/' + dataset + '_' + strategy + '/test2id.txt') as f:
                    line = f.readline()
                    for line in f:
                        h,r,t = line.split('\t')
                        h = int(h)
                        t = int(t)
                        entity_set.add(h)
                        entity_set.add(t)
                        count += 1
                        if count == count_nums:
                            break
                return list(entity_set)

            con.test(get_test_entity(20000),20000)

test('2')

#con.show_link_prediction(2,1)
#con.show_triple_classification(2,1,3)
# (2) Read model parameters from json files and manually load parameters.
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# f = open("./res/embedding.vec.json", "r")
# content = json.loads(f.read())
# f.close()
# con.set_parameters(content)
# con.test()

# (3) Manually load models via tf.Saver().
# con = config.Config()
# con.set_in_path("./benchmarks/FB15K/")
# con.set_test_flag(True)
# con.set_work_threads(4)
# con.set_dimension(50)
# con.init()
# con.set_model(models.TransE)
# con.import_variables("./res/model.vec.tf")
# con.test()

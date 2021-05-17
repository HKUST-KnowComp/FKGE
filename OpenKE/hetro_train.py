import sys
import OpenKE.config as config
import OpenKE.models as models
import tensorflow as tf
import numpy as np
import os
import json

check_files = ['entity2id.txt', 'relation2id.txt', 'train2id.txt', 'valid2id.txt', 'test2id.txt',
               '1-1.txt', '1-n.txt', 'n-1.txt', 'n-n.txt', 'test2id_all.txt']


def train(exp_id, mode, epoch_num, dimension_num, kg, cur_round, embedding_path, federated):
    origin_kg_path = './OpenKE/benchmarks/' + kg + '_1/'
    extended_kg_path = './experiment/' + str(exp_id) + '/' + kg + '/extended/'

    if mode == 'strategy_2' and federated:
        kg_path = extended_kg_path
    else:
        kg_path = origin_kg_path
    
    # validation check: check if target .txt file exists and recorded object number coincide with actual object number
    for check_file in check_files:
        if not os.path.exists(os.path.join(kg_path, check_file)):
            raise Exception('Error! %s of %s does not exist.' % (check_file, kg_path))
            sys.exit()
        
        with open(os.path.join(kg_path, check_file)) as fp:
            record_obj_num = int(fp.readline().strip())
            actual_obj_num = len(fp.readlines())
            if check_file == 'entity2id.txt':
                actual_entity_num = actual_obj_num
        if record_obj_num != actual_obj_num:
            raise Exception('Error! recorded object number of %s in %s does not coincide with actual object number(%d != %d).' % (check_file, kg_path, record_obj_num, actual_obj_num))
            sys.exit()

    con = config.Config()

    con.set_in_path(kg_path)
    # con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)
    con.set_work_threads(8)
    con.set_train_times(epoch_num)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(dimension_num)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")

    export_path = './experiment/' + str(exp_id) + '/' + kg + '/model/' + str(cur_round)
    new_embedding_path = export_path + '/embedding.json'
    os.makedirs(export_path, exist_ok=True)

    con.set_export_files(export_path + '/model.vec.tf', 0)
    con.set_out_files(export_path + '/embedding.json')
    
    # Initialize experimental settings.
    con.init()
    
    # Set the knowledge embedding model
    con.set_model(models.TransE, kg)

    # load post-trained embedding
    if embedding_path is not None:
        if federated:
            model_data = np.load('./experiment/' + str(exp_id) + '/' + kg + '/GAN_files/' + kg + '_embedding.npy')
        else:
            with open(embedding_path, 'r') as fp:    
                model_data = json.load(fp)
                model_data = model_data['ent_embeddings']
            
        # validation check: check if post-trained entity number coincide with actual entity number
        posttrain_entity_num = len(model_data)
        if posttrain_entity_num != actual_entity_num:
            print('Error! %s post-trained entity number does not coincide with actual entity number(%d != %d).' % (kg_path, posttrain_entity_num, actual_entity_num))
        con.set_parameters_by_name('ent_embeddings', model_data)

    # Train the model.
    con.run()

    return con.test(), new_embedding_path

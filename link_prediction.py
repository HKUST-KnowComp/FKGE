import sys
import os
import json

from OpenKE.origin_config.Config import Config
from OpenKE.hetro_train import check_files
import OpenKE.models as models
from log import Logit

def find_best_round(task_map, target_id):
    all_kg = task_map[target_id]['all_kg']
    best_round_map = dict()

    for kg in all_kg:
        now_id = int(target_id)
        while now_id != -1:
            for round in os.listdir(os.path.join('experiment', str(now_id), kg, 'model')):
                embedding_path = os.path.join('experiment', str(now_id), kg, 'model', round, 'embedding.json')
                if os.path.exists(embedding_path) and os.path.getsize(embedding_path) > 16:
                    if best_round_map.get(kg) == None or int(round) > best_round_map.get(kg)[1]:
                        best_round_map[kg] = (now_id, int(round))
            if best_round_map.get(kg):
                break
            else:
                now_id = task_map[str(now_id)]["kg2pred_id"][kg]
    if len(best_round_map) != len(all_kg):
        wrong_kgs = filter(lambda kg, best_round_map: best_round_map.get(kg) == None, all_kg)
        wrong_kgs = list(wrong_kgs)
        raise Exception('thers exists kg in all_Kg that doesn`t have best round model: {}'.format(wrong_kgs))
    return best_round_map

def test_link_prediction(best_round_map, task_map, kg):
    kg_path  = './OpenKE/benchmarks/' + kg + '_1/'
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
            if check_file == 'relation2id.txt':
                actual_relation_num = actual_obj_num
        if record_obj_num != actual_obj_num:
            raise Exception('Error! recorded object number of %s in %s does not coincide with actual object number(%d != %d).' % (check_file, kg_path, record_obj_num, actual_obj_num))

    con = Config()

    con.set_in_path(kg_path)
    con.set_test_link_prediction(True)
    con.set_work_threads(8)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(task_map[str(best_round_map[kg][0])]['dimension_num'])
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)

    # Initialize experimental settings.
    con.init()
    
    # Set the knowledge embedding model
    con.set_model(models.TransE, kg)

    # load post-trained embedding
    embedding_path = os.path.join('experiment', str(best_round_map[kg][0]), kg, 'model', str(best_round_map[kg][1]), 'embedding.json')
    with open(embedding_path, 'r') as fp:
        print('json load {}'.format(embedding_path))   
        model_data = json.load(fp)
        ent_embeddings = model_data['ent_embeddings'][ : actual_entity_num]
        rel_embeddings = model_data['rel_embeddings'][ : actual_relation_num]
    
    con.set_parameters_by_name('ent_embeddings', ent_embeddings)
    con.set_parameters_by_name('rel_embeddings', rel_embeddings)

    con.test()

def test_triple_classification(kg, kg_path, embedding_path):
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
            if check_file == 'relation2id.txt':
                actual_relation_num = actual_obj_num
        if record_obj_num != actual_obj_num:
            raise Exception('Error! recorded object number of %s in %s does not coincide with actual object number(%d != %d).' % (check_file, kg_path, record_obj_num, actual_obj_num))

    con = Config()

    con.set_in_path(kg_path)
    con.test_triple_classification(True)
    con.set_work_threads(8)
    con.set_nbatches(100)
    con.set_alpha(0.001)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(task_map[str(best_round_map[kg][0])]['dimension_num'])
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)

    # Initialize experimental settings.
    con.init()
    
    # Set the knowledge embedding model
    con.set_model(models.TransE, kg)

    # load post-trained embedding
    with open(embedding_path, 'r') as fp:
        print('json load {}'.format(embedding_path))   
        model_data = json.load(fp)
        ent_embeddings = model_data['ent_embeddings'][ : actual_entity_num]
        rel_embeddings = model_data['rel_embeddings'][ : actual_relation_num]
    
    con.set_parameters_by_name('ent_embeddings', ent_embeddings)
    con.set_parameters_by_name('rel_embeddings', rel_embeddings)

    return con.test()

if __name__ == "__main__":
    target_id = sys.argv[1]
    kg = sys.argv[2]

    task_map = Logit.get_task_map()['tasks']

    # key = kg : str, value = (task_id : int, round_num : int)
    best_round_map = find_best_round(task_map, target_id)
    print(best_round_map)

    test_link_prediction(best_round_map, task_map, kg)
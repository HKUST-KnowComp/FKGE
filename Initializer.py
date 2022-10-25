import os
import json
from log import Logit

class Initializer:
    '''
    sementic predecessor
        no replicant baseline with same epoch_num
        pred_id must be -1 or a valid task
    attribute
        self.pred_id
        self.pred_round
        self.pred_result
        self.pred_embedding_path
    '''
    def __init__(self, kg, succ_mode, epoch_num, dimension, pred_id):
        self.log_map = Logit.get_task_map()['tasks']
        
        pred_id = int(pred_id)
        if succ_mode == 'baseline':
            if pred_id == -1:
                # baseline start from 0
                self.pred_id = -1
                self.pred_embedding_path = None
            else:
                # baseline succeeding baseline, from previous best
                self.recursive_load(succ_mode, kg, pred_id)
                        
        elif succ_mode == 'strategy_1' or succ_mode == 'strategy_2':
            if self.log_map[str(pred_id)]['mode'] == 'baseline':
                # strategy succeeding baseline, from start
                self.pred_id = pred_id
                
                with open(os.path.join('experiment', str(pred_id), 'result', kg + '.txt')) as fp:
                    line = fp.readline().strip()
                    line = line.replace('*', '')
                    line = line[line.find(']') + 2:]
                    #self.pred_round = int(line.split(' ')[0][:-1])
                    self.pred_round = int(line.split(' ')[0])
                    self.pred_result = float(line.split(' ')[1])
                self.pred_embedding_path = os.path.join('experiment', str(pred_id), kg, 'model', str(self.pred_round), 'embedding.json')
            elif self.log_map[str(pred_id)]['mode'] == 'strategy_1' or self.log_map[str(pred_id)]['mode'] == 'strategy_2':
                if kg not in self.log_map[str(pred_id)]['all_kg']:
                    # scablable, adding new kg, from baseline start
                    for task_id in self.log_map.keys():
                        if self.log_map[task_id]['mode'] == 'baseline' and int(self.log_map[task_id]['epoch_num']) == epoch_num and int(self.log_map[task_id]['dimension_num']) == dimension:
                            break
                    print('Scablable new kg %s succeeding baseline %s' % (kg, task_id))
                    self.pred_id = int(task_id)

                    with open(os.path.join('experiment', str(task_id), 'result', kg + '.txt')) as fp:
                        line = fp.readline().strip()
                        line = line[line.find(']') + 2:]
                        self.pred_round = int(line.split(' ')[0])
                        self.pred_result = float(line.split(' ')[1])
                    self.pred_embedding_path = os.path.join('experiment', str(task_id), kg, 'model', str(self.pred_round), 'embedding.json')
                else:
                    # strategy succeeding strategy, from previous best
                    self.recursive_load(succ_mode, kg, pred_id)
            else:
                raise Exception('Error! Task %d has wrong mode %s' % (pred_id, self.log_map[pred_id['mode']]))
        else:
            raise Exception('Error! Sucessor has wrong mode')

    def recursive_load(self, succ_mode, kg, pred_id):
        self.pred_round = None
        while succ_mode == self.log_map[str(pred_id)]['mode'] and not self.pred_round:
            with open(os.path.join('experiment', str(pred_id), 'result', kg + '.txt')) as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) == 0 or 'Federated' in line or '*' not in line:
                        continue
                    line = line[line.find(']') + 2:]
                    self.pred_round = int(line.split(' ')[0][:-1])
                    self.pred_result = float(line.split(' ')[1])
            if not self.pred_round:
                if int(self.log_map[str(pred_id)]['kg2pred_id'][kg]) == -1:
                    break
                else:
                    pred_id = int(self.log_map[str(pred_id)]['kg2pred_id'][kg])
                
 
        if not self.pred_round:
            print('Can\'t find previous best round of %s, using round 0 of baseline.' % kg)
            with open(os.path.join('experiment', str(pred_id), 'result', kg + '.txt')) as fp:
                line = fp.readline().strip()
                line = line[line.find(']') + 2:]
                self.pred_round = int(line.split(' ')[0])
                self.pred_result = float(line.split(' ')[1])
            self.pred_embedding_path = os.path.join('experiment', str(pred_id), kg, 'model', str(self.pred_round), 'embedding.json')
        
        self.pred_id = pred_id
        self.pred_embedding_path = os.path.join('experiment', str(pred_id), kg, 'model', str(self.pred_round), 'embedding.json')

import os
import sys
import time
import enum
import json
import shutil
import argparse
import traceback
import numpy as np
from multiprocessing import Manager, Pool
from hetro_init import init_file
from trainse_data.function import add_node
from OpenKE.hetro_train import train
from OpenKE.benchmarks.n_n import constrain
from OpenKE.benchmarks.change_graph import change_graph
from FederalTransferLearning.hetro_AGCN_mul_dataset import GAN
from log import Logit
from Initializer import Initializer


start_time = time.time()


class Config:
    def __init__(self, exp_id, mode, epoch_num, dimension_num, gan_ratio, all_kg, initializer):
        self.exp_id = exp_id
        self.mode = mode
        self.epoch_num = epoch_num
        self.dimension_num = dimension_num
        self.gan_ratio = gan_ratio
        self.all_kg = all_kg
        self.initializer = initializer
        
        # get origin KG size
        self.kg2ori_size = dict()
        for kg in all_kg:
            path = './OpenKE/benchmarks/' + kg + '_1/entity2id.txt'
            with open(path, 'r') as fp:
                self.kg2ori_size[kg] = int(fp.readline().strip())

        # create experiment folder, in which all related files will be stored
        exp_folder = os.path.join('./experiment', str(exp_id))
        os.makedirs(os.path.join(exp_folder, 'result'), exist_ok=True)
        for kg in all_kg:
            kg_folder = os.path.join(exp_folder, kg)
            model_folder = os.path.join(kg_folder, 'model')
            os.makedirs(model_folder, exist_ok=True)
        self.exp_folder = exp_folder

        # create data structure for Federated Learning
        if mode == 'strategy_1' or mode == 'strategy_2':
            self.kg2connected = dict()
            self.gan_queue = Manager().dict()
            self.gan_kg2round = Manager().dict()
            self.lock = Manager().Lock()

            for kg in all_kg:
                self.kg2connected[kg] = set()
                self.gan_queue[kg] = json.dumps(list())
                self.gan_kg2round[kg] = json.dumps(dict())

            # add KG connection
            aligned_path = './trainse_data/aligned'
            for f in os.listdir(aligned_path):
                f = f[:-4]
                source_kg = f.split('_')[0]
                target_kg = f.split('_')[1]
                if source_kg in all_kg and target_kg in all_kg:
                    self.kg2connected[source_kg].add(target_kg)
                    self.kg2connected[target_kg].add(source_kg)
    
    def __str__(self):
        return json.dumps(self.__repr__())
    
    def __repr__(self):
        task = dict()
        task['task_id'] = self.exp_id
        task['mode'] = self.mode
        task['epoch_num'] = self.epoch_num
        task['dimension_num'] = self.dimension_num
        task['gan_ratio'] = self.gan_ratio
        task['all_kg'] = self.all_kg
        kg2init_id = dict()
        for kg, init in self.initializer.items():
            kg2init_id[kg] = init.pred_id
        if len(kg2init_id) != len(self.all_kg):
            raise Exception('Error: wrong kg2pred_id len compared whith all_kg')
        task['kg2pred_id'] = kg2init_id
        return task
    
    def record(self, kg, cur_round, result, improve):
        cur_time = time.time()
        result_dir = os.path.join(self.exp_folder, 'result')
        with open(os.path.join(result_dir, kg + '.txt'), 'a+') as fp:
            fp.write('[' + str(cur_time - start_time) + ']' + ' ')
            fp.write(str(cur_round))
            if improve:
                fp.write('*')
            fp.write(' ' + str(result) + '\n')

    @Logit()
    def make_gan(self, receive, receive_embedding_path, send, send_embedding_path, encrypter=None):
        # check if align .npy exists
        align_link_path = './trainse_data/aligned/' + send + '_' + receive + '.npy'
        if not os.path.exists(align_link_path):
            print('Error! No aligned entities between %s abd %s' %s (send, receive))
            sys.exit()
        
        # create GAN_file
        GAN_folder = os.path.join(self.exp_folder, receive, 'GAN_files')
        os.makedirs(GAN_folder, exist_ok=True)

        # load embeddings and link
        with open(receive_embedding_path, 'r') as fp:
            receive_data = json.load(fp)
            receive_embedding = np.array(receive_data['ent_embeddings'])
        with open(send_embedding_path, 'r') as fp:
            send_data = json.load(fp)
            send_embedding = np.array(send_data['ent_embeddings'])
        align_link = np.load(align_link_path)
        align_link = align_link[:int(len(align_link) * self.gan_ratio)]

        if self.mode == 'strategy_2':
            # calculate 1-setp nodes
            send_edge, send_add_node, send_link_node = add_node(send, align_link)

            # dump 1-step nodes
            with open(os.path.join(GAN_folder, send + '_anode_link.json'), 'w') as fp:
                json.dump(send_link_node, fp)

        # construct align embedding respectively
        receive_align_embedding = receive_embedding[align_link[:,1].astype('int32'),:]
        send_align_embedding = send_embedding[align_link[:,0].astype('int32'),:]

        if self.mode == 'strategy_2':
            send_align_embedding = np.concatenate((send_align_embedding, send_embedding[list(send_add_node),:]))

        # save align embedding respectively
        np.save(os.path.join(GAN_folder, receive + '_align_embedding.npy'), receive_align_embedding)
        np.save(os.path.join(GAN_folder, send + '_align_embedding.npy'), send_align_embedding)
    
    @Logit()
    def make_embedding(self, first, receive, receive_embedding_path, send, encrytper=None):
        # check if align .npy exists
        receive_send_addr = './trainse_data/aligned/' + receive + '_' + send + '.npy'
        if not os.path.exists(receive_send_addr):
            print('Error! No aligned entities between %s and %s.' % (receive, send))
            sys.exit()

        # load embedding, GAN result and align .npy
        if first:
            with open(receive_embedding_path, 'r') as fp:
                receive_data = json.load(fp)
                receive_embedding = np.array(receive_data['ent_embeddings'])
                receive_embedding = receive_embedding[:self.kg2ori_size[receive]]
        else:
            receive_embedding = np.load(os.path.join(self.exp_folder, receive, 'GAN_files', receive + '_embedding.npy'))
        align_embedding = np.load(os.path.join(self.exp_folder, receive, 'GAN_files', receive + '_gan_embedding.npy'))
        receive_send = np.load(receive_send_addr)
        receive_send = receive_send[:int(len(receive_send) * self.gan_ratio)]

        # replace aligned nodes with GAN result
        receive_embedding[receive_send[:,0],:] += align_embedding[0:receive_send.shape[0]]
        receive_embedding[receive_send[:,0],:] /= 2

        if self.mode == 'strategy_2':
            # concatenate anodes
            receive_embedding = np.concatenate((receive_embedding, align_embedding[receive_send.shape[0]:]))

        # save replaced embedding as the initiation of next round training
        np.save(os.path.join(self.exp_folder, receive, 'GAN_files', receive + '_embedding.npy'), receive_embedding)

    def baseline(self, kg):
        try:
            if self.initializer[kg].pred_id == -1:
                cur_round = 0
                best_round = None
                best_result = None
                best_embedding_path = None
            else:
                cur_round = self.initializer[kg].pred_round + 1
                best_round = self.initializer[kg].pred_round
                best_result = self.initializer[kg].pred_result
                best_embedding_path = self.initializer[kg].pred_embedding_path
            
            improve = False
            while True:
                result, new_embedding_path = train(self.exp_id, self.mode, self.epoch_num, self.dimension_num, kg, cur_round, best_embedding_path, False)

                if best_round is None:
                    improve = True
                    best_round = cur_round
                    best_result = result
                elif result > best_result:
                    improve = True
                    best_round = cur_round
                    best_result = result
                    best_embedding_path = new_embedding_path
                else:
                    improve = False
                    model_dir = os.path.join(self.exp_folder, kg, 'model', str(cur_round))
                    shutil.rmtree(model_dir)
                self.record(kg, cur_round, result, improve)
                if improve == True:
                    cur_round += 1
        except BaseException as exc:
            m = traceback.format_exc()
            print(m)
            Logit.notify(m)
            sys.exit()
    
    def federated(self, kg):
        try:
            best_round = self.initializer[kg].pred_round
            best_result = self.initializer[kg].pred_result
            best_embedding_path = self.initializer[kg].pred_embedding_path

            result_file = os.path.join(self.exp_folder, 'result', kg + '.txt')
            while(True):
                # send signal to connected kg
                for connected_kg in self.kg2connected[kg]:
                    self.lock.acquire()
                    gan_queue_tmp = json.loads(self.gan_queue[connected_kg])
                    gan_kg2round_tmp = json.loads(self.gan_kg2round[connected_kg])

                    if kg not in gan_queue_tmp:
                        gan_queue_tmp.append(kg)
                    gan_kg2round_tmp[kg] = best_embedding_path

                    self.gan_queue[connected_kg] = json.dumps(gan_queue_tmp)
                    self.gan_kg2round[connected_kg] = json.dumps(gan_kg2round_tmp)
                    self.lock.release()
        
                # check queue, get picture, wait if empty
                while True:
                    self.lock.acquire()
                    gan_queue_tmp = json.loads(self.gan_queue[kg])
                    gan_kg2round_tmp = json.loads(self.gan_kg2round[kg])

                    # flush queue
                    self.gan_queue[kg] = json.dumps(list())
                    self.lock.release()
                    
                    if not gan_queue_tmp:
                        time.sleep(100)
                    else:
                        break

                ############### Federate Module ###############
                # record Federated status
                cur_time = time.time()
                with open(result_file, 'a+') as fp:
                    fp.write('[' + str(cur_time - start_time) + ']' + ' ')
                    if gan_queue_tmp:
                        fp.write('Federated with: ')
                        for send in gan_queue_tmp:
                            fp.write(send + ' ')
                        fp.write('\n')
                    else:
                        fp.write('Unfederated\n')
                
                # GAN one by one, change graph if strategy_2
                first = True
                for send in gan_queue_tmp:
                    send_embedding_path = gan_kg2round_tmp[send]
                    self.make_gan(kg, best_embedding_path, send, send_embedding_path)
                    GAN(self.exp_id, kg, send, self.dimension_num)
                    self.make_embedding(first, kg, best_embedding_path, send)
                    first = False
                
                if self.mode == 'strategy_2':
                    change_graph(self.exp_id, kg, gan_queue_tmp)
                    constrain(0, self.exp_id, kg)
                ############### Federate Module ###############

                ############### Train Module ###############
                improve = False
                
                cur_round = best_round + 1
                result, new_embedding_path = train(self.exp_id, self.mode, self.epoch_num, self.dimension_num, kg, cur_round, best_embedding_path, True)
                if result > best_result:
                    improve = True
                    best_round = cur_round
                    best_result = result
                    best_embedding_path = new_embedding_path
                else:
                    model_dir = os.path.join(self.exp_folder, kg, 'model', str(cur_round))
                    shutil.rmtree(model_dir)
                    improve = False
                self.record(kg, cur_round, result, improve)
                ############### Train Module ###############
        except BaseException as exc:
            m = traceback.format_exc()
            print(m)
            Logit.notify(m)
            sys.exit()
 
    def run(self):
        process = Pool(len(self.all_kg))
        if self.mode == 'baseline':
            for kg in self.all_kg:
                process.apply_async(func=self.baseline, args=[kg])
        elif self.mode == 'strategy_1' or self.mode == 'strategy_2':
            for kg in self.all_kg:
                process.apply_async(func=self.federated, args=[kg])
        process.close()
        process.join()


if __name__ == '__main__':
    Logit.init_task_id()
    exp_id = Logit.task_id
    mode = sys.argv[1]
    epoch_num = int(sys.argv[2])
    dimension_num = int(sys.argv[3])
    gan_ratio = float(sys.argv[4])
    pred_id = int(sys.argv[5])

    all_kg = ['geonames', 'yago', 'dbpe', 'poke', 'geospecies', 'sandrart', 'police', 'lex', 'tharawat', 'whisky', 'worldlift']
    
    # scalable_1
    # all_kg = ['geonames', 'sandrart']
    
    # scalable_2
    # all_kg = ['geonames', 'sandrart', 'geospecies']

    # scalable_3
    # all_kg = ['geonames', 'sandrart', 'geospecies', 'worldlift']

    # scalable_4
    # all_kg = ['geonames', 'sandrart', 'geospecies', 'worldlift', 'lex', 'tharawat']

    # all_in_one
    # all_kg = ['AIO']
    
    # split_geo
    # all_kg = ['subgeonamesA', 'subgeonamesB']
    # init_file(all_kg)

    initializer = dict()
    for kg in all_kg:
        initializer[kg] = Initializer(kg, mode, epoch_num, dimension_num, pred_id)

    con = Config(exp_id, mode, epoch_num, dimension_num, gan_ratio, all_kg, initializer)
    Logit.update_task_map(con)
    con.run()

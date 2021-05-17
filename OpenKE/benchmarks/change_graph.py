import numpy as np
import os
import sys
import json
import shutil


def change_graph(exp_id, receive, gan_sequence):
    base_path = os.path.join('experiment', str(exp_id))
    origin_path_dir = './OpenKE/benchmarks/' + receive + '_1/'
    path_dir = os.path.join(base_path, receive, 'extended/')

    os.makedirs(path_dir, exist_ok= True)

    # clear path_dir
    files = ['entity2id.txt', 'relation2id.txt', 'train2id.txt', 'valid2id.txt', 'test2id.txt']
    for file in files:
        if os.path.exists(path_dir + file):
            os.remove(path_dir + file)
        shutil.copy(origin_path_dir + file, path_dir + file)

    receive_path = os.path.join(base_path, receive)
    for send in gan_sequence:
        if not os.path.exists(os.path.join(receive_path, 'GAN_files', send + '_anode_link.json')):
            raise Exception('Error! Anode link %s in %s not created.' % (send, receive))

        with open(os.path.join(receive_path, 'GAN_files', send + '_anode_link.json'), 'r') as f:
            link_node = json.load(f)
        add_node = len(link_node.keys())
        receive_send = np.load('./trainse_data/aligned/' + send + '_' + receive + '.npy').astype('int32')
        receive_send_dict = {}
        for i in range(receive_send.shape[0]):
            receive_send_dict[str(receive_send[i, 0])] = receive_send[i, 1]

        with open(path_dir + 'entity2id.txt', 'r+', encoding='utf-8') as f:
            entity_old_num = int(f.readline().strip())

            old = f.read()
            f.seek(0)
            f.write(str(entity_old_num + add_node) + '\n')
            f.write(old)
            if old[-1] != '\n':
                f.write('\n')

            for i in range(add_node - 1):
                f.write('new_node\t' + str(i + entity_old_num) + '\n')
            f.write('new_node\t' + str(add_node - 1 + entity_old_num))

        with open(path_dir + 'relation2id.txt', 'r+', encoding='utf-8') as f:
            relation_old_num = int(f.readline().strip())

            old = f.read()
            f.seek(0)
            f.write(str(relation_old_num + 1) + '\n')
            f.write(old)
            if old[-1] != '\n':
                f.write('\n')

            f.write('new_relation_' + send + '\t' + str(relation_old_num))

        link_node_list = list(link_node.keys())
        l = len(link_node_list)

        with open(path_dir + 'train2id.txt', 'r+') as f:
            train_old_num = int(f.readline().strip())
            print('train2id_old_num:', train_old_num)

            old = f.read()
            f.seek(0)
            f.write(str(train_old_num + l) + '\n')
            f.write(old)
            if old[-1] != '\n':
                f.write('\n')

            for i in range(l - 1):
                f.write(str(entity_old_num) + '\t' + str(receive_send_dict[link_node_list[i]]) + '\t' + str(relation_old_num) + '\n')
                entity_old_num += 1
            f.write(str(entity_old_num) + '\t' + str(receive_send_dict[link_node_list[l-1]]) + '\t' + str(relation_old_num))
            entity_old_num += 1
        
        '''
        We think there is no need for the project to change valid2id.txt file
        with open(path_dir + 'valid2id.txt', 'r+') as f:
            valid_old_num = int(f.readline().strip())
            print('valid2id_old_num:', valid_old_num)

            old = f.read()
            f.seek(0)
            f.write(str(valid_old_num + l - l) + '\n')
            f.write(old)
            if old[-1] != '\n':
                f.write('\n')

            for i in range(l, l - 1):
                f.write(str(entity_old_num) + '\t' + str(receive_send_dict[link_node_list[i]]) + '\t' + str(relation_old_num) + '\n')
                entity_old_num += 1
            f.write(str(entity_old_num) + '\t' + str(receive_send_dict[link_node_list[l1-1]]) + '\t' + str(relation_old_num))
            entity_old_num += 1
        '''
        
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
# from OpenKE.hetro_train import train
# from OpenKE.benchmarks.n_n import constrain
# from OpenKE.benchmarks.change_graph import change_graph
# from FederalTransferLearning.hetro_AGCN_mul_dataset import GAN
from log import Logit
from Initializer import Initializer


start_time = time.time()


def make_gan(receive, receive_embedding_path, send, send_embedding_path, encrypter=None):
    # check if align .npy exists
    align_link_path = './trainse_data/aligned/' + send + '_' + receive + '.npy'
    if not os.path.exists(align_link_path):
        print('Error! No aligned entities between %s abd %s' %s (send, receive))
        sys.exit()
    
    # create GAN_file
    exp_folder=os.path.join('./0_experiment', '0')
    GAN_folder = os.path.join(exp_folder, receive, 'GAN_files')
    os.makedirs(GAN_folder, exist_ok=True)

    # load embeddings and link
    with open(receive_embedding_path, 'r') as fp:
        receive_data = json.load(fp)
        receive_embedding = np.array(receive_data['ent_embeddings'])
    with open(send_embedding_path, 'r') as fp:
        send_data = json.load(fp)
        send_embedding = np.array(send_data['ent_embeddings'])
    align_link = np.load(align_link_path)
    align_link = align_link[:int(len(align_link) * 1)]


    # construct align embedding respectively
    receive_align_embedding = receive_embedding[align_link[:,1].astype('int32'),:]
    send_align_embedding = send_embedding[align_link[:,0].astype('int32'),:]

    # save align embedding respectively
    np.save(os.path.join(GAN_folder, receive + '_align_embedding.npy'), receive_align_embedding)
    np.save(os.path.join(GAN_folder, send + '_align_embedding.npy'), send_align_embedding)
receive_embedding_path=os.path.join('all_experiment', '0', 'geospecies', 'model', str(0), 'embedding.json')
send_embedding_path=os.path.join('all_experiment', '0', 'geonames', 'model', str(0), 'embedding.json')
make_gan('geospecies', receive_embedding_path, 'geonames', send_embedding_path, encrypter=None)   
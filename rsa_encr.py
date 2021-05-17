import os
import sys
import timeit
import shutil
from multiprocessing import Manager, Pool
from hetro_init import init_file_1
from OpenKE.hetro_train import train
from trainse_data.hetro_make_embedding_1 import make_embedding
from trainse_data.hetro_make_gan_1 import make_gan
from FederalTransferLearning.hetro_AGCN_mul_dataset import GAN
from encrypt.encrypt import RsaEncrypt as Encrypter


def record(kg, cur_round, database):
    if cur_round == 0:
        database[kg]['best_round'] = 0
        database[kg]['best_result'] = database[kg]['result']

        result_dir = './result/plaintext/' + kg + '.txt'
        with open(result_dir, 'a+') as f:
            f.write(str(best_round) + ' ' + str(best_result) + '\n')
    elif database[kg]['result'] > database[kg]['best_result']:
        # update
        database[kg]['best_round'] = cur_round
        database[kg]['best_result'] = database[kg]['result']

        result_dir = './result/plaintext/' + kg + '.txt'
        with open(result_dir, 'a+') as f:
            f.write(str(best_round) + '* ' + str(best_result) + '\n')
    else:
        result_dir = './result/plaintext/' + kg + '.txt'
        with open(result_dir, 'a+') as f:
            f.write(str(cur_round) + ' ' + str(database[kg]['result']) + '\n')
        model_dir = './OpenKE/benchmarks/' + kg + '_1/model/' + str(cur_round)
        shutil.rmtree(model_dir)


def federate(receive, send, database):
    encrypter = Encrypter()
    encrypter.generate_key(rsa_bit=32)
    make_gan(receive, database[receive]['best_round'], send, database['send']['best_round'], encrypter)
    GAN(1, receive, send)
    make_embedding(True, receive, database[receive]['best_round'], send, encrypter)


def run(all_kg, database):
    cur_round = 0
    federated = False
    while(True):
        # train embedding
        process = Pool(len(all_kg))
        result_list = [result.get() for result in [process.apply_async(func=train, args=[1, kg, cur_round, federated, database[kg]['best_round']]) for kg in all_kg]]
        process.close()
        process.join()

        for i, kg in enumerate(all_kg):
            database[kg]['result'] = result_list[i]
            print(kg, database[kg]['result'])

        # record result
        process = Pool(len(all_kg))
        for kg in all_kg:
            process.apply_async(func=record, args=[kg, cur_round, database])
        process.close()
        process.join()

        # federate
        process = Pool(len(all_kg))
        for kg in all_kg:
            if kg == 'dbpe':
                receive = 'dbpe'
                send = 'yago'
            else:
                receive = 'yago'
                send = 'dbpe'
            process.apply_async(func=federate, args=[receive, send, database])
        federated = True
        cur_round += 1
        if cur_round == 50:
            break


if __name__ == '__main__':
    all_kg = ['dbpe', 'yago']

    # init file
    #init_file_1(all_kg)
    result_dir = './result'
    plaintext_dir = './result/plaintext'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(plaintext_dir):
        os.mkdir(plaintext_dir)
    
    # init data structure
    database = dict()
    for kg in all_kg:
        database[kg] = dict()
        database[kg]['result'] = None
        database[kg]['best_round'] = None
        database[kg]['best_result'] = None   

    run(all_kg, database)


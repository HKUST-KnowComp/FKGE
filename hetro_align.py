import os
import numpy as np


def init_data(data, kg):
    print('Loading %s entities...' % (kg))
    tmp = dict()
    tmp['map'] = dict()
    entity2id_addr = './OpenKE/benchmarks/' + kg + '_1/entity2id.txt'
    entity2id = open(entity2id_addr, 'r', encoding='utf-8')
    entity2id.readline()
    for line in entity2id:
        line = line.replace('\n', '')
        entity = line.split('\t')[0]
        id = line.split('\t')[1]
        tmp['map'][entity] = id
    entity2id.close()

    data[kg] = dict()
    data[kg] = tmp


def remove_unexpected_align(data, kg):
    print('Removing %s unexpected align...' % (kg))
    intersection_addr = './OpenKE/benchmarks/' + kg + '_1/intersection'
    for f in os.listdir(intersection_addr):
        if '2' in f:
            old_addr = intersection_addr + '/' + f
            new_addr = intersection_addr + '/' + 'new_' + f
            new_line_num = 0
            with open(old_addr, 'r', encoding='utf-8') as old_file:
                with open(new_addr, 'w', encoding='utf-8') as new_file:
                    old_file.readline()
                    for line in old_file:
                        entity = line.split('\t')[0]
                        if entity in data['map'].keys():
                            new_line_num += 1
                            new_file.write(line)
            with open(new_addr, 'r+', encoding='utf-8') as f:
                content = f.read()
                f.seek(0)
                f.write(str(new_line_num) + '\n')
                f.write(content)
            os.system('mv' + ' ' + new_addr + ' ' + old_addr) 


def generate_npy_single(data_source, data_target, source_kg, target_kg):
    print('Generating .npy for %s and %s(single)...' % (source_kg, target_kg))
    intersection_addr = './OpenKE/benchmarks/' + source_kg + '_1/intersection/' + source_kg + '2' + target_kg + '.txt'
    intersection_file = open(intersection_addr, 'r', encoding='utf-8')

    source_target = np.empty(shape=[0, 2], dtype=np.int32)
    target_source = np.empty(shape=[0, 2], dtype=np.int32)
    intersection_file.readline()
    in_count = 0
    not_in_count = 0
    for line in intersection_file:
        line = line.replace('\n', '')
        source_entity = line.split('\t')[0]
        target_entity = line.split('\t')[1]
        if source_entity not in data_source['map'].keys():
            print('Error! Entity %s from %s in intersection file not found in entity2id.txt.' % (source_entity, source_kg))
            sys.exit()
        if target_entity in data_target['map'].keys():
            in_count += 1
            source_id = data_source['map'][source_entity]
            target_id = data_target['map'][target_entity]
            source_target = np.append(source_target, np.array([[source_id, target_id]], dtype=np.int32), axis=0)
            target_source = np.append(target_source, np.array([[target_id, source_id]], dtype=np.int32), axis=0)
        else:
            not_in_count += 1
    print('%s to %s aligned entities: %d. Not aligned entities: %d.' % (source_kg, target_kg, in_count, not_in_count))
    if in_count:
        print('Saving %s_%s...' % (source_kg, target_kg))
        source_target_npy = './trainse_data/aligned/' + source_kg + '_' + target_kg + '.npy'
        target_source_npy = './trainse_data/aligned/' + target_kg + '_' + source_kg + '.npy'
        np.save(source_target_npy, source_target)
        np.save(target_source_npy, target_source)
    else:
        print('No entities aligned from %s to %s.' % (source_kg, target_kg))
    record_addr = './align_record/' + source_kg + '2' + target_kg + '.txt'
    with open(record_addr, 'w', encoding='utf-8') as f:
        f.write('%d\t%d' % (in_count, not_in_count))


def generate_npy_union(data_kg1, data_kg2, kg_1, kg_2):
    print('Generating .npy for %s and %s(union)...' % (kg_1, kg_2))
    intersection_1_addr = './OpenKE/benchmarks/' + kg_1 + '_1/intersection/' + kg_1 + '2' + kg_2 + '.txt'
    intersection_2_addr = './OpenKE/benchmarks/' + kg_2 + '_1/intersection/' + kg_2 + '2' + kg_1 + '.txt'
    intersection_1_file = open(intersection_1_addr, 'r', encoding='utf-8')
    intersection_2_file = open(intersection_2_addr, 'r', encoding='utf-8')

    kg1_kg2 = np.empty(shape=[0, 2], dtype=np.int32)
    kg2_kg1 = np.empty(shape=[0, 2], dtype=np.int32)
    intersection_1_file.readline()
    intersection_2_file.readline()

    total_count = 0
    in_count = 0
    not_in_count = 0
    for line in intersection_1_file:
        line = line.replace('\n', '')
        kg1_entity = line.split('\t')[0]
        kg2_entity = line.split('\t')[1]
        if kg1_entity not in data_kg1['map'].keys():
            print('Error! Entity %s from %s in intersection file not found in entity2id.txt.' % (kg1_entity, kg_1))
            sys.exit()
        if kg2_entity in data_kg2['map'].keys():
            total_count += 1
            in_count += 1
            kg1_id = data_kg1['map'][kg1_entity]
            kg2_id = data_kg2['map'][kg2_entity]
            kg1_kg2 = np.append(kg1_kg2, np.array([[kg1_id, kg2_id]], dtype=np.int32), axis=0)

            kg2_kg1 = np.append(kg2_kg1, np.array([[kg2_id, kg1_id]], dtype=np.int32), axis=0)
        else:
            not_in_count += 1
    print('%s to %s aligned entities: %d. Not aligned entities: %d.' % (kg_1, kg_2, in_count, not_in_count))
    record_addr = './align_record/' + kg_1 + '2' + kg_2 + '.txt'
    with open(record_addr, 'w', encoding='utf-8') as f:
        f.write('%d\t%d' % (in_count, not_in_count))

    in_count = 0
    not_in_count = 0
    for line in intersection_2_file:
        line = line.replace('\n', '')
        kg2_entity = line.split('\t')[0]
        kg1_entity = line.split('\t')[1]
        if kg2_entity not in data_kg2['map'].keys():
            print('Error! Entity %s from %s in intersection file not found in entity2id.txt.' % (kg2_entity, kg_2))
        if kg1_entity in data_kg1['map'].keys():
            total_count += 1
            in_count += 1
            kg2_id = data_kg2['map'][kg2_entity]
            kg1_id = data_kg1['map'][kg1_entity]
            kg1_kg2 = np.append(kg1_kg2, np.array([[kg1_id, kg2_id]], dtype=np.int32), axis=0)
            kg2_kg1 = np.append(kg2_kg1, np.array([[kg2_id, kg1_id]], dtype=np.int32), axis=0)
        else:
            not_in_count += 1
    print('%s to %s aligned entities: %d. Not aligned entities: %d.' % (kg_2, kg_1, in_count, not_in_count))
    if total_count:
        print('Saving %s_%s...' % (kg_1, kg_2))
        dst_1_addr = './trainse_data/aligned/' + kg_1 + '_' + kg_2 + '.npy'
        dst_2_addr = './trainse_data/aligned/' + kg_2 + '_' + kg_1 + '.npy'
        np.save(dst_1_addr, kg1_kg2)
        np.save(dst_2_addr, kg2_kg1)
    else:
        print('No entities aligned between %s and %s.' % (kg_1, kg_2))
    record_addr = './align_record/' + kg_2 + '2' + kg_1 + '.txt'
    with open(record_addr, 'w', encoding='utf-8') as f:
        f.write('%d\t%d' % (in_count, not_in_count))


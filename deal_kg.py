'''
trainse_data/aligned  aligned vectice, npy file, ndarray, ele is [source id, target id]
OpenKE/benchmarks/{kg name}_1 
    all file in this dir has a special first line which indicate the amount of data in same file
    entity2id.txt line is "enetity name \t id"
    train2id
    test2id.txt line is "eid\t eid\t edge id"
    valid2id.txt
'''
import os

record_file_name_list = ['train2id.txt', 'test2id.txt', 'valid2id.txt']
clip_kg_names =  ['geonames_1', 'dbpe_1', 'yago_1']
in_edge = 0
out_edge = 1

def can_clip(e1, entity_dict, aligned_set):
    can_clip_b = True
    if e1 in aligned_set:
        can_clip_b = False
    for edge, e2, _ in entity_dict[e1]:
        if e2 in aligned_set:
            can_clip_b = False
            break
    return can_clip_b

def clip_kg(kg_path):
    kg_name = kg_path.split('/')[-1][:-2]
    kg_str = ''
    for file_name in record_file_name_list:
        with open(os.path.join(kg_path, file_name), 'r') as f:
            kg_str += '\n'
            kg_str += f.read()


    entity_dict = {}
    for i, line in enumerate(kg_str.split('\n')):
        line = line.strip()
        if len(line) == 0 or i == 0:
            continue
        line = [line.split('\t')]
        e1, e2, edge = line[0], line[1], line[2]
        entity_dict.setdefault(e1, set()).add((edge, e2, out_edge))
        entity_dict.setdefault(e2, set()).add((edge, e1, in_edge))
    sorted_list = sorted(entity_dict.items(),key = lambda item : len(item[1]))
    
    aligned_set = set()
    for aligned_file_name in os.list_dir(os.path.join('trainse_data', 'aligned'))
        if aligned_file_name.split('_')[0] != kg_name:
            continue
        aligned_arr = np.load(os.path.join('trainse_data', 'aligned', aligned_file_name))
        for i in aligned_arr.shape[0]:
            aligned_set.add(aligned_arr[i][0])
    most_clip_amount = int(0.8 * len(entity_dict))
    clip_amount = 0
    del_list = []
    for i, item in enumerate(sorted_list):
        if not can_clip(item[0]):
            continue
        if clip_amount >= most_clip_amount:
            break
        clip_amount += 1
        entity_dict.pop(item[0])
    if clip_amount < most_clip_amount:
        for item in sorted_list:
            if clip_amount >= most_clip_amount:
                break
            if item[0] in entity2id:
                clip_kg_names += 1
                entity_dict.pop(item[0])
    out_list = []
    for e, rela_set in entity_dict.values():
        for rela in rela_set:
            if rela[1] in entity_dict and rela[2] == out_edge:
                out_list.append((e, rela[1], rela[0]))
    train_index = int(len(out_list) * 0.9)
    test_index = int(len(out_list) * 0.95)
    with open(os.path.join(kg_path, 'train2id.txt'), 'w') as f:
        w_str = str(train_index)
        for item in out_list[:train_index]:
            w_str += '\n' + str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2])
        f.write(w_str)
    with open(os.path.join(kg_path, 'test2id.txt'), 'w') as f:
        w_str = str(test_index - train_index)
        for item in out_list[train_index : test_index]:
            w_str += '\n' + str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2])
        f.write(w_str)
    with open(os.path.join(kg_path, 'valid2id.txt'), 'w') as f:
        w_str = str(len(out_list) - test_index)
        for item in out_list[test_index : ]:
            w_str += '\n' + str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2])
        f.write(w_str)

if __name__ == '__main__':
    for path, dir_list, file_name_list in os.walk('OpenKE/benchmarks'):
        if path[-1] != '1':
            continue
        elif path.split('/')[-1] in clip_kg_names:
            clip_kg(path)

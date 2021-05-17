import os
from random import shuffle


base_path = './OpenKE/benchmarks'


def split(edges, kg, part, ele2old_map):
    ent_map = dict()
    rel_map = dict()
    ele2new_list = dict()
    new_edges = list()
    for ele in ['entity', 'relation']:
        ele2new_list[ele] = list()
    
    for i in range(part[0], part[1]):
        ent_1 = int(edges[i][0])
        ent_2 = int(edges[i][1])
        rel = int(edges[i][2])
        for ent in [ent_1, ent_2]:
            if ent not in ent_map.keys():
                ent_map[ent] = len(ent_map.keys())
                ele2new_list['entity'].append(ele2old_map['entity'][ent])
        if rel not in rel_map.keys():
            rel_map[rel] = len(rel_map.keys())
            ele2new_list['relation'].append(ele2old_map['relation'][rel])
        new_edges.append((ent_map[ent_1], ent_map[ent_2], rel_map[rel]))
    
    for ele in ['entity', 'relation']:
        with open(os.path.join(base_path, kg + '_1', ele + '2id.txt'), 'w') as fp:
            fp.write(str(len(ele2new_list[ele])))
            for index, element in enumerate(ele2new_list[ele]):
                fp.write('\n' + element + '\t' + str(index))
    
    train_num = int(0.9 * len(new_edges))
    valid_num = int(0.05 * len(new_edges))
    test_num = len(new_edges) - train_num - valid_num
    for index, edge in enumerate(new_edges):
        if index == 0:
            with open(os.path.join(base_path, kg + '_1', 'train2id.txt'), 'w') as fp:
                fp.write(str(train_num))
        elif 0 < index < train_num:
            with open(os.path.join(base_path, kg + '_1', 'train2id.txt'), 'a+') as fp:
                fp.write('\n' + str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(edge[2]))
        elif index == train_num:
            with open(os.path.join(base_path, kg + '_1', 'valid2id.txt'), 'w') as fp:
                fp.write(str(valid_num))
        elif train_num < index < train_num + valid_num:
            with open(os.path.join(base_path, kg + '_1', 'valid2id.txt'), 'a+') as fp:
                fp.write('\n' + str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(edge[2]))
        elif index == train_num + valid_num:
            with open(os.path.join(base_path, kg + '_1', 'test2id.txt'), 'w') as fp:
                fp.write(str(test_num))
        else:
            with open(os.path.join(base_path, kg + '_1', 'test2id.txt'), 'a+') as fp:
                fp.write('\n' + str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(edge[2]))


if __name__ == '__main__':
    kg = 'geonames'
    kg_path = os.path.join(base_path, kg + '_1')

    # load entities and relations indexed by id
    ele2old_map = dict()
    for ele in ['entity', 'relation']:
        id2element = dict()
        with open(os.path.join(kg_path, ele + '2id.txt'), 'r') as fp:
            fp.readline()
            for line in fp.readlines():
                line = line.strip()
                element = line.split('\t')[0]
                id = int(line.split('\t')[1])
                id2element[id] = element
        ele2old_map[ele] = id2element

    # load edges
    edges = list()
    for ele in ['train', 'valid', 'test']:
        with open(os.path.join(kg_path, ele + '2id.txt'), 'r') as fp:
            fp.readline()
            for line in fp.readlines():
                line = line.strip()
                ent_1 = int(line.split('\t')[0])
                ent_2 = int(line.split('\t')[1])
                rel = int(line.split('\t')[2])
                edges.append((ent_1, ent_2, rel))

    # shuffle edges
    shuffle(edges)

    # split edges by half
    part_1 = (0, len(edges) // 2)
    part_2 = (len(edges) // 2, len(edges))

    # create dirs
    new_kg_1 = 'sub' + kg + 'A'
    new_kg_2 = 'sub' + kg + 'B'

    os.makedirs(os.path.join(base_path, new_kg_1 + '_1', 'intersection'), exist_ok=True)
    os.makedirs(os.path.join(base_path, new_kg_2 + '_1', 'intersection'), exist_ok=True)

    # record aligned entities
    new_ent_1 = set()
    new_ent_2 = set()
    align = list()
    for i in range(len(edges)):
        ent_1 = edges[i][0]
        ent_2 = edges[i][1]
        if i in range(0, len(edges) // 2):
            new_ent_1.add(ent_1)
            new_ent_1.add(ent_2)
        else:
            new_ent_2.add(ent_1)
            new_ent_2.add(ent_2)
    for ent in new_ent_1:
        if ent in new_ent_2:
            align.append(ent)
    
    with open(os.path.join(base_path, new_kg_1 + '_1', 'intersection', new_kg_1 + '2' + new_kg_2 + '.txt'), 'w') as fp:
        fp.write(str(len(align)))
        for ent in align:
            fp.write('\n' + ele2old_map['entity'][ent] + '\t' + ele2old_map['entity'][ent])
    with open(os.path.join(base_path, new_kg_2 + '_1', 'intersection', new_kg_2 + '2' + new_kg_1 + '.txt'), 'w') as fp:
        fp.write(str(len(align)))
        for ent in align:
            fp.write('\n' + ele2old_map['entity'][ent] + '\t' + ele2old_map['entity'][ent])

    # split kg, rehash entities, relations, edges
    split(edges, new_kg_1, part_1, ele2old_map)
    split(edges, new_kg_2, part_2, ele2old_map)

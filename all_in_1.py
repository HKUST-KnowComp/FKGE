import os


all_kg = ['geonames', 'yago', 'dbpe', 'poke', 'geospecies', 'sandrart', 'police', 'lex', 'tharawat', 'whisky', 'worldlift']
new_kg = 'AIO'
base_path = './OpenKE/benchmarks'

# create dir
os.makedirs(os.path.join(base_path, new_kg + '_1', 'intersection'), exist_ok=True)

# concate each kg
ele2list = dict()
for ele in ['entity', 'relation', 'train', 'valid', 'test']:
    ele2list[ele] = list()

for kg in all_kg:
    for ele in ['train', 'test', 'valid']:
        with open(os.path.join(base_path, kg + '_1', ele + '2id.txt'), 'r') as fp:
            fp.readline()
            for line in fp.readlines():
                line = line.strip()
                ent_1 = len(ele2list['entity']) + int(line.split('\t')[0])
                ent_2 = len(ele2list['entity']) + int(line.split('\t')[1])
                rel = len(ele2list['relation']) + int(line.split('\t')[2])
                ele2list[ele].append((str(ent_1), str(ent_2), str(rel)))
    
    for ele in ['entity', 'relation']:
        with open(os.path.join(base_path, kg + '_1', ele + '2id.txt'), 'r') as fp:
            fp.readline()
            for line in fp.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                element = line.split('\t')[0]
                ele2list[ele].append(element)

# write file
for ele in ['entity', 'relation']:
    with open(os.path.join(base_path, new_kg + '_1', ele + '2id.txt'), 'w') as fp:
        fp.write(str(len(ele2list[ele])))
        for index, element in enumerate(ele2list[ele]):
            fp.write('\n' + element + '\t' + str(index))

for ele in ['train', 'test', 'valid']:
    with open(os.path.join(base_path, new_kg + '_1', ele + '2id.txt'), 'w') as fp:
        fp.write(str(len(ele2list[ele])))
        for element in ele2list[ele]:
            fp.write('\n' + element[0] + '\t' + element[1] + '\t' + element[2])

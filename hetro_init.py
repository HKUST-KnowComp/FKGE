import os
import sys
from multiprocessing import Manager, Pool
from OpenKE.benchmarks.n_n import constrain
from hetro_align import init_data, remove_unexpected_align, generate_npy_single, generate_npy_union


def init_file(all_kg):    
    # generate type constrain
    process = Pool(len(all_kg))
    for kg in all_kg:
        process.apply_async(constrain, args=[1, None, kg])
    process.close()
    process.join()
    
    # classify align condition
    print('Classifying align condition...')
    single = set()
    union = set()
    for kg in all_kg:
        path = './OpenKE/benchmarks/' + kg + '_1/intersection'
        for f in os.listdir(path):
            if '2' in f and 'new_' not in f:
                f = f[:-4]
                source_kg = f.split('2')[0]
                target_kg = f.split('2')[1]
                if not source_kg == kg:
                    print('Error with kg naming! Registered as %s but saved as %s for intersection files.' % (kg, source_kg))
                    sys.exit()
                if target_kg in all_kg:
                    source_index = all_kg.index(source_kg)
                    target_index = all_kg.index(target_kg)
                    target_path = './OpenKE/benchmarks/' + target_kg + '_1/intersection/' + target_kg + '2' + kg + '.txt'
                    if os.path.exists(target_path):
                        tup = (min(source_index, target_index), max(source_index, target_index))
                        union.add(tup)
                    else:
                        single.add((source_index, target_index))
    total = single | union
    single_num = len(single)
    union_num = len(union)
    total_num = len(total)
    if not total_num == single_num + union_num:
        print('Error with align condition classification! Some kgs are both singly and unionly aligned!')
        sys.exit()

    data = Manager().dict()
    # load entity2id into data
    process = Pool(len(all_kg))
    for kg in all_kg:
        process.apply_async(init_data, args=(data, kg,))
    process.close()
    process.join()

    # remove unexpected align entities
    process = Pool(len(all_kg))
    for kg in all_kg:
        process.apply_async(remove_unexpected_align, args=(data[kg], kg,))
    process.close()
    process.join()

    # generate align .npy files
    process = Pool(total_num)
    for (source_index, target_index) in single:
        source_kg = all_kg[source_index]
        target_kg = all_kg[target_index]
        process.apply_async(generate_npy_single, args=(data[source_kg], data[target_kg], source_kg, target_kg,))
    for (kg1_index, kg2_index) in union:
        kg1 = all_kg[kg1_index]
        kg2 = all_kg[kg2_index]
        process.apply_async(generate_npy_union, args=(data[kg1], data[kg2], kg1, kg2,))
    process.close()
    process.join()

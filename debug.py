import os
from Parser import Parser
from log import Logit


task_map = Logit.get_task_map()['tasks']
for task_id in task_map.keys():
    if int(task_id) < 100 or task_map[task_id]['mode'] != 'baseline':
        continue

    result_path = os.path.join('experiment', task_id, 'result')
    for kg in task_map[task_id]['all_kg']:
        old_file_path = os.path.join(result_path, kg + '.txt')
        with open(old_file_path, 'r') as fp_old:
            for line in fp_old:
                round = line.split(' ')[1].replace('*', '')
        if round == 0:
            continue
        
        new_file_path = os.path.join(result_path, 'new' + kg + '.txt')
        with open(old_file_path, 'r') as fp_old:
            with open(new_file_path, 'w') as fp_new:
                for line in fp_old:
                    line = line.split(' ')[0] + ' ' + line.split(' ')[1].replace(round, str(int(round) - 1)) + ' ' + line.split(' ')[2]
                    fp_new.write(line)
        os.system('mv ' + new_file_path + ' ' + old_file_path)


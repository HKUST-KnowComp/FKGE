import os


class Parser:
    def __init__(self, exp_id):
        self.exp_id = exp_id
        self.result = dict()
        self.parse()
    
    def parse(self):
        result_path = './experiment/' + str(self.exp_id) + '/result/'
        for file in os.listdir(result_path):
            kg = file[:-4]
            self.result[kg] = list()
            
            with open(result_path + file, 'r', encoding='utf-8') as fp:
                while True:
                    line = fp.readline().strip()
                    if len(line) == 0:
                        break
                    
                    record = dict()
                    if 'F' in line:
                        if '[' in line and ']' in line:
                            record['gan_time'] = float(line.split(' ')[0][1:-1])
                        else:
                            record['gan_time'] = None
                        record['gan_list'] = line.split(':')[1].strip().split(' ')
                        line = fp.readline().strip()
                        if len(line) == 0:
                            break
                    else:
                        record['gan_time'] = None
                        record['gan_list'] = None
                    
                    if '*' in line:
                        record['improve'] = True
                        line = line.replace('*', '')
                    else:
                        record['improve'] = False

                    if '[' in line and ']' in line:
                        record['time'] = float(line.split(' ')[0][1:-1])
                        record['round'] = int(line.split(' ')[1])
                        record['accuracy'] = float(line.split(' ')[2])
                    else:
                        record['time'] = None
                        record['round'] = int(line.split(' ')[0])
                        record['accuracy'] = float(line.split(' ')[1])

                    self.result[kg].append(record)

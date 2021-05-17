import os
import time
import json
from functools import wraps

kg_set = set(['geonames', 'yago', 'dbpe', 'poke', 'geospecies', 'sandrart', 'police', 'lex', 'tharawat', 'whisky', 'worldlift'])

class Logit:
    '''
    the logit system class
    all mothod of Logit class should be called in the way of static for the correct running of logit system
    '''
    
    task_id = 0
    base_path = 'experiment'
    logit_path = None
    task_map_path = os.path.join(base_path, 'task_map.txt')

    def __init__(self):
        pass
    
    @classmethod
    def init_task_id(cls):
        '''
        this method must be called before the logit system beginning work
        it will read the task_map.txt to get a unique task_id and update logit_map.txt
        '''

        os.makedirs(cls.base_path, exist_ok= True)
        if not os.path.exists(os.path.join(cls.base_path, 'task_map.txt')):
            with open(os.path.join(cls.base_path, 'task_map.txt'), 'w') as f:
                # task_id is beginning from 0
                task_map = dict()
                task_map['task_id'] = 0
                task_map['tasks'] = dict()
                f.write(json.dumps(task_map, indent= 4, separators= (',', ':')))
                cls.task_id = 0
        else:
            with open(os.path.join(cls.base_path, 'task_map.txt'), 'r') as f:
                # get task_id
                task_map = json.load(f)
                cls.task_id = task_map['task_id'] + 1
                task_map['task_id'] = cls.task_id
            with open(os.path.join(cls.base_path, 'task_map.txt'), 'w') as f:
                # update task_id
                f.write(json.dumps(task_map, indent= 4, separators= (',', ':')))
        cls.logit_path = os.path.join(Logit.base_path, str(cls.task_id), 'logit')
        os.makedirs(cls.logit_path,  exist_ok= True)
    
    @classmethod
    def update_task_map(cls, config):
        '''
        this method must be called after the calling of init_task_id method
        it will record current config information to the end of task_map.txt
        '''

        with open(cls.task_map_path, 'r') as f:
            task_map = json.load(f)
            if task_map['tasks'].get(str(Logit.task_id)) != None:
                raise Exception('duplicated defined task_id')
            task_map['tasks'][str(cls.task_id)] = str(config)
        with open(cls.task_map_path, 'w') as f:
            f.write(json.dumps(task_map, indent= 4, separators= (',', ':')))

    @classmethod
    def get_task_map(cls):
        with open(cls.task_map_path, 'r') as f:
            task_map = json.load(f)
            for id in task_map['tasks'].keys():
                task_map['tasks'][id] = json.loads(task_map['tasks'][id])
        return task_map

    def __call__(self, func):
        '''
        this is a decorator method, it should be used as following:
            @Logit()
            func_be_decorated:
                func_body_codes
        '''

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            Logit.check_system_status()

            # check whether there is element of cur_kg_list in args or kwargs

            kg = None
            for arg in args:
                if arg in kg_set:
                    kg = arg
                    break
            for arg in kwargs.values():
                if arg in kg_set:
                    kg = arg
                    break
            
            if kg != None:
                # logit in kg logit file
                log_str = Logit.get_time()
                log_str += 'enter {:<20}, kg {}'.format(getattr(func, '__name__'), kg)
                with open(os.path.join(self.logit_path, kg + '.txt'), 'a+') as f:
                    f.write(log_str + '\n')
            # logit in task logit file
            log_str = Logit.get_time()

            log_str += 'pid{:<10} enter func {}'.format(os.getpid(), getattr(func, '__name__'))
            with open(os.path.join(self.logit_path, 'task_logit.txt'), 'a+') as f:
                f.write(log_str + '\n')
            return func(*args, **kwargs)
        return wrapped_func
    
    @classmethod
    def get_time(cls):
        '''
        get current time in special format as year-month-day-hours:minutes:seconds
        '''

        local_time = time.localtime(time.time())
        return  '${:>5}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}$ '.format(str(local_time[0]), str(local_time[1]),
                str(local_time[2]), str(local_time[3]), str(local_time[4]), str(local_time[5]))
    
    @classmethod
    def notify(cls, messege):
        '''
        if something monitered and important has happened, write messege to notify.txt file
        '''

        cls.check_system_status()
        with open(os.path.join(cls.logit_path, 'notify.txt'), 'a+') as f:
            f.write('\n===========================================================\n')
            f.write(Logit.get_time() + '\n')
            f.write('===========================================================\n')
            f.write(messege)
    
    @classmethod
    def check_system_status(cls):
        '''
        check the status of Logit system when it is calling to work
        '''

        if cls.logit_path == None:
            raise Exception('Logit class has not been prepared well')
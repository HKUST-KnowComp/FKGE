from enum import Enum
import json
import os

import torch
from torch.utils.data import DataLoader

from KGEmbedding.codes.dataloader import TrainDataset
from KGEmbedding.codes.dataloader import BidirectionalOneShotIterator
from KGEmbedding.codes.model import KGEModel

class Config:
    class Mode(Enum):
        TransE = 'TransE'
        DistMult = 'DistMult'
        # this two algorithms isn`t matched with other algorithms in the hidden dimension
        #ComplEx = 'ComplEx'
        #RotatE = 'RotatE'
        pRotatE = 'pRotatE'
    
    def __init__(self):
        self.max_steps = 100000
        self.warm_up_steps = None
        self.ent_embeddings = None
    
    def set_mode(self, mode):
        self.mode = mode

    def set_train_times(self, max_steps):
        self.max_steps = max_steps
    
    def set_dimension(self, dimension):
        self.dimension = dimension
    
    def set_in_path(self, kg_path):
        self.kg_path = kg_path
    
    def set_ent_embeddings(self, ent_embeddings):
        self.ent_embeddings = ent_embeddings
    
    def set_save_path(self, save_path):
        self.save_path = save_path
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_work_threads(self, work_threads_num):
        self.work_threads_num = work_threads_num
    
    def set_negative_sample_size(self, negative_sample_size):
        self.negative_sample_size = negative_sample_size

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    
    def set_warm_up_steps(self, warm_up_steps):
        self.warm_up_steps = warm_up_steps
        if self.warm_up_steps > self.max_steps or self.warm_up_steps <= 0:
            raise ValueError('Error: {} has a wrong value {}'.format(getattr(self.warm_up_steps, '__name__'),
             self.warm_up_steps))
    
    def init(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        with open(os.path.join(self.kg_path, 'entity2id.txt')) as fin:
            self.nentity = int(fin.readline())
        
        with open(os.path.join(self.kg_path, 'relation2id.txt')) as fin:
            self.nrelation = int(fin.readline())
        
        self.train_triples = self.read_triple(os.path.join(self.kg_path, 'train2id.txt'))
        self.valid_triples = self.read_triple(os.path.join(self.kg_path, 'valid2id.txt'))
        self.test_triples = self.read_triple(os.path.join(self.kg_path, 'test2id.txt'))

        self.all_true_triples = self.train_triples + self.valid_triples + self.test_triples

        self.kg_model =  KGEModel(
            model_name= self.mode.value,
            nentity= self.nentity,
            nrelation= self.nrelation,
            hidden_dim= self.dimension,
            gamma= self.gamma,
            device= self.device,
            double_entity_embedding= False,
            double_relation_embedding= False
        )
        
        if self.ent_embeddings:
            self.kg_model.set_embedding(self.ent_embeddings, None)
        
        '''
        if torch.cuda.device_count() > 1:
            print('model run in {} GPUs'.format(torch.cuda.device_count()))
            self.kg_model = torch.nn.DataParallel(self.kg_model)
        '''
        self.kg_model.to(self.device)
    
    def run(self):
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(self.train_triples, self.nentity, self.nrelation, self.negative_sample_size,
                 'head-batch'), 
            batch_size= self.batch_size,
            shuffle= True, 
            num_workers= self.work_threads_num,
            collate_fn= TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(self.train_triples, self.nentity, self.nrelation, self.negative_sample_size,
                 'tail-batch'), 
            batch_size= self.batch_size,
            shuffle= True, 
            num_workers= self.work_threads_num,
            collate_fn= TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = self.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kg_model.parameters()), 
            lr= current_learning_rate
        )
        if self.warm_up_steps:
            warm_up_steps = self.warm_up_steps
        else:
            warm_up_steps = self.max_steps // 2
        
        valid_steps = self.max_steps // 10 if self.max_steps > 100 else 10
        for step in range(0, self.max_steps):
            print('step {}'.format(step))
            log = self.kg_model.train_step(self.kg_model, optimizer, train_iterator)
            for k, v in log.items():
                print(k, v)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                print('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.kg_model.parameters()), 
                    lr= current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % valid_steps == 0:
                print('Evaluating on Valid Dataset...')
                metrics = self.kg_model.test_step(self.kg_model, self.valid_triples, self.all_true_triples, self)
                print('Valid', step, 'MPR', metrics['MPR'])
        
        embeddings = self.kg_model.get_embedding()
        embeddings = {
            'ent_embeddings' : embeddings[0],
            'rel_embeddings' : embeddings[1]
        }

        self.save_model(embeddings)
    
    def test(self):
        print('Evaluating on Test Dataset...')
        metrics = self.kg_model.test_step(self.kg_model, self.test_triples, self.all_true_triples, self)
        print('Valid', step, 'MPR', metrics['MPR'])
        return metrics['MPR']

    def read_triple(self, file_path):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        with open(file_path) as fin:
            ntriples = int(fin.readline())
            for line in fin.readlines():
                h, t, r = line.strip().split('\t')
                triples.append((int(h), int(r), int(t)))
            if ntriples != len(triples):
                raise Exception('Error, ntriples({}) is not matched with the size of triples list({})'.
                    format(ntriples, line(triples)))
        return triples
        
    def save_model(self, model_dict):
        with open(self.save_path, 'w') as f:
            f.write(json.dumps(model_dict))
        
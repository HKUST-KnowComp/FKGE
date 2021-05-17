import tensorflow as tf
import numpy as np
import time
import csv
import sys
import os

#lap var
import tensorflow_probability as tfp
fold = 10

# GCN
fc_output_size = 1024

#train
batch_size = 32
epoch_num = 1
learning_rate = 1e-2
momentum = 0.9

x = None
fake_x = None
x_size = None
fake_x_size = None
data_size = None
test_size = None
train_size = None
train_index = None
train_gan_label = None
test_index = None
test_gan_label = None


def init(exp_id, receive, send):
    global x, fake_x, fake_x_size, data_size, test_size, train_size, train_index, train_gan_label, test_index, test_gan_label
    
    x = np.load('./experiment/' + str(exp_id) + '/' + receive + '/GAN_files/' + receive + '_align_embedding.npy')
    fake_x = np.load('./experiment/' + str(exp_id) + '/' + receive + '/GAN_files/' + send + '_align_embedding.npy')
    x_size = x.shape[0]
    fake_x_size = fake_x.shape[0]
    data_size = x_size + fake_x_size
    test_size = int(data_size / fold)
    train_size = data_size - test_size
    train_index, train_gan_label, test_index, test_gan_label = read_data()


def read_data():
    index = [i for i in range(data_size)]
    np.random.shuffle(index)
    gan_label = np.zeros((data_size))
    gan_label[:x_size] = gan_label[:x_size] + 1
    gan_label = gan_label[index]
    return index[:train_size], gan_label[:train_size], index[train_size:], gan_label[train_size:]


def get_data(ix, int_batch):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    batch_gan_label = train_gan_label[ix:end]
    batch_index = train_index[ix:end]
    return batch_index, batch_gan_label


class AGCN(object):
    def __init__(self, session,
                 data_size,
                 fc_output_size,
                 embedding):
        self.data_size = data_size
        self.embedding = embedding
        self.fc_output_size = fc_output_size
        self.teacher_num = 4
        
        self.build_placeholders()
        
        self.loss_g, self.loss_d, self.loss, self.probabilities, self.foo = self.forward_propagation()
        one = tf.ones_like(self.probabilities)
        zero = tf.zeros_like(self.probabilities)
        self.pred = tf.where(self.probabilities<0.5, x=zero, y=one)
        correct_prediction = tf.equal(self.pred, self.gan_t)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('Forward propagation finished.')
         
        self.sess = session

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = self.optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)

        self.init = tf.global_variables_initializer()
        print('Backward propagation finished.')
        
    def build_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, self.embedding], 'x')
        self.fake_x = tf.placeholder(tf.float32, [None, self.embedding], 'fake_x')
        self.index = tf.placeholder(tf.int32, [None], 'index')
        self.gan_t = tf.placeholder(tf.float32, [None], 'gan_labels') # [1,1,1,1,...,0,0,0,0]
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.placeholder(tf.float32, [], 'momentum')
        

    def forward_propagation(self):
        with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name='g_weights', shape=[self.embedding, self.embedding], initializer=tf.contrib.layers.xavier_initializer())
            fake_out = tf.matmul(self.fake_x, W)
            out = tf.concat([self.x, fake_out], 0)
        #hr added for teacher D
        
        ###divide index for teacher
        num_index = tf.cast(tf.shape(self.index)[0] / self.teacher_num, tf.int32)
        t_losses = []
        g_losses = []
        t_preds = []
        loss_list = []
        for i in range(self.teacher_num):
            cur_index = self.index[i*num_index:i*num_index+num_index]
            cur_t = self.gan_t[i*num_index:i*num_index+num_index]
            cur_out = tf.matmul(tf.one_hot(cur_index, self.data_size), out)
            total_loss, pred = self.forward_t(current_tid=i,sep_out=cur_out,sep_t=cur_t)
            #t_losses.append(loss_t)
            #g_losses.append(loss_g)
            #t_preds.append(pred)
            loss_list.append(total_loss)
            
        ### implementation for pate mechanism
        for i in range(self.teacher_num):
            cur_index = self.index
            cur_t = self.gan_t
            cur_out = tf.matmul(tf.one_hot(cur_index, self.data_size), out)
            total_loss, pred = self.forward_t(current_tid=i,sep_out=cur_out,sep_t=cur_t)
            t_preds.append(pred)
        loss= tf.add_n(loss_list)  #current loss for teachers and generators
        LAP = tfp.distributions.Laplace(0.0,0.05)
        noise = LAP.sample(sample_shape=(tf.shape(t_preds)))
        agg_pred = (tf.add_n(t_preds) + noise) / self.teacher_num
        agg_one = tf.ones_like(agg_pred)
        agg_zero = tf.zeros_like(agg_pred)
        agg_pred = tf.where(agg_pred<0.5, x=agg_one, y=agg_zero)
        
        
        with tf.variable_scope('student_discriminator',reuse=tf.AUTO_REUSE):
            shuffled_data = tf.matmul(tf.one_hot(self.index, self.data_size), out)
            fc1 = tf.layers.dense(inputs=shuffled_data, units=self.fc_output_size, activation=None)
            fc2 = tf.layers.dense(inputs=fc1, units=self.fc_output_size, activation=None)
            fc4 = tf.layers.dense(inputs=fc2, units=1, activation=None) # probability of true
            fc3 = tf.nn.sigmoid(fc4)
            fc3 = tf.reshape(fc3, (-1,))
            loss_d = -tf.reduce_mean(tf.log(1e-8 + tf.multiply(1-fc3, 1-agg_pred))) - tf.reduce_mean(tf.log(1e-8 + tf.multiply(fc3, agg_pred)))
            loss_g = -tf.reduce_mean(tf.log(1e-8 + tf.multiply(fc3, 1-agg_pred))) - tf.reduce_mean(tf.log(1e-8 + tf.multiply(1-fc3, agg_pred)))

        with tf.variable_scope('classification'):
            loss += loss_g + loss_d

        return loss_g, loss_d, loss, fc3, fake_out
    
    #hr added for teacher D forward
    def forward_t(self,current_tid,sep_out,sep_t):
        #reuse scope
        with tf.variable_scope('teacher_discriminator_'+str(current_tid),reuse=tf.AUTO_REUSE):
            
            #D layers
            fc1 = tf.layers.dense(inputs=sep_out, units=self.fc_output_size, activation=None)
            fc2 = tf.layers.dense(inputs=fc1, units=self.fc_output_size, activation=None)
            fc4 = tf.layers.dense(inputs=fc2, units=1, activation=None) # probability of true
            fc3 = tf.nn.sigmoid(fc4)
            fc3 = tf.reshape(fc3, (-1,))
            loss_d = -tf.reduce_mean(tf.log(1e-8 + tf.multiply(1-fc3, 1-sep_t))) - tf.reduce_mean(tf.log(1e-8 + tf.multiply(fc3, sep_t)))
            #loss_g = -tf.reduce_mean(tf.log(1e-8 + tf.multiply(fc3, 1-sep_t))) - tf.reduce_mean(tf.log(1e-8 + tf.multiply(1-fc3,sep_t)))
            loss =  loss_d
            
            return loss,fc3
        
    def train(self, x, fake_x, ix, gt, learning_rate = 1e-3, momentum = 0.9):
        feed_dict = {
            self.x: x,
            self.fake_x: fake_x,
            self.index: ix,
            self.gan_t: gt,
            self.lr: learning_rate,
            self.mom: momentum
        }
        _, loss, acc, pred, foo = self.sess.run([self.train_op, self.loss, self.accuracy, self.pred, self.foo], feed_dict = feed_dict)
        
        return loss, acc, pred, foo

    def test(self, x, fake_x, ix, gt):
        feed_dict = {
            self.x: x,
            self.fake_x: fake_x,
            self.index: ix,
            self.gan_t: gt
        }
        acc, pred = self.sess.run([self.accuracy, self.pred], feed_dict = feed_dict)
        return acc, pred
    
    
    

      
def com_f1(pred,label):
    MI_F1 = []
    l = len(pred)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    f1 = 0
    for i in range(l):
        if pred[i] == 1 and label[i] == 1:
            TP += 1
        elif pred[i] == 1:
            FP += 1
        elif label[i] == 1:
            FN += 1
        else:
            TN += 1
    if TP+FP == 0:
       pre = 0
    else:
       pre = TP/(TP + FP)
    if TP+FN == 0:
       rec = 0
    else:
       rec = TP/(TP + FN)
    acc = (TP+TN)/l
    if (pre + rec) != 0:
        f1 = 2*pre*rec/(pre+rec)
    return [pre,rec,acc,f1]


def GAN(exp_id, receive, send, embedding_dimension):
    tf.reset_default_graph()
    init(exp_id, receive, send)
    exit_count = 0
    early_loss = 0

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            if receive == 'subgeonamesA':
                device = '/device:GPU:0'
            elif receive == 'geonames' or receive == 'worldlift' or receive == 'whisky' or receive == 'tharawat' or receive == 'lex':
                device = '/device:GPU:0'
            else:
                device = '/device:GPU:1'
            with tf.device(device):
                net = AGCN(session=sess, data_size=data_size, fc_output_size=fc_output_size, embedding= embedding_dimension)
                sess.run(tf.global_variables_initializer())

                min_loss = 15061162
                max_acc = -1
                loss_upper_bound = 100
                for epoch in range(epoch_num):
                    train_loss = 0
                    train_acc = 0
                    count = 0
                    
                    for index in range(0, train_size, batch_size):
                        batch_index, batch_gan_label = get_data(index, batch_size)
                        loss, acc, pred, foo = net.train(x, fake_x, batch_index, batch_gan_label, learning_rate, momentum)
                        if loss == early_loss:
                            exit_count += 1
                        else:
                            early_loss = loss

                        if index % 1 == 0:
                            print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
                        train_loss += loss
                        train_acc += acc
                        count += 1
                        np.save('./experiment/' + str(exp_id) + '/' + receive + '/GAN_files/' + receive + '_gan_embedding.npy', foo)
                        if exit_count == 5:
                            return
                    train_loss = train_loss/count
                    train_acc = train_acc/count
                    if train_loss < min_loss:
                        min_loss = train_loss
                    print("--------------------------------------------------------------")
                    print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, train_loss, train_acc))
                    print("--------------------------------------------------------------")
                    eva_acc, eva_pred = net.test(x, fake_x, test_index, test_gan_label)
                    with open('./experiment/' + str(exp_id) + '/' + receive + '/GAN_files/train_acc.txt', 'a+') as f:
                        f.write(str(train_acc))
                        f.write('\n')
                    with open('./experiment/' + str(exp_id) + '/' + receive + '/GAN_files/test_acc.txt', 'a+') as f:
                        f.write(str(eva_acc))
                        f.write('\n')
                    with open('./experiment/' + str(exp_id) + '/' + receive + '/GAN_files/train_loss.txt', 'a+') as f:
                        f.write(str(train_loss))
                        f.write('\n')

                    if eva_acc > max_acc:
                        max_acc = eva_acc
                        print('present max accuracy:', eva_acc)
                        print('golden label:', test_gan_label)
                        print('pred label:', eva_pred)
                        print('********************* Model Saved *********************')
        print("Train end!")
        print("The loss is {:.4f}, the acc is {:.4f}".format(min_loss, max_acc))

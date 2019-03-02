import os
import argparse
import pickle
import numpy as np
from keras.layers import Conv1D, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils, Sequence
import datetime
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='neural net training script')
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = base_path + '/../data'
    exp_path = base_path + '/../exp_results'
    parser.add_argument('-d', '--decode', action='store_true',
                        help='true in the decoding phase')
    parser.add_argument('-decode-net-file', default=exp_path + '/1/trained_models/iter4.h5',
                        help='path for the network in the decode phase')
    parser.add_argument('-decode-out-dir', default=exp_path + '/decode',
                        help='path for the decoding results')
    parser.add_argument('-out-dir', default=exp_path,
                        type=str, help='output directory for this experiment')
    parser.add_argument('-tr-file', default=data_path + '/train_si284.pickle',
                        type=str, help='train pickle file path')
    parser.add_argument('-de-file', default=data_path + '/test_dev93.pickle',
                        type=str, help='development pickle file path')
    parser.add_argument('-te-file', default=data_path + '/test_eval93.pickle',
                        type=str, help='test pickle file path')
    parser.add_argument('-utt-in-dim', default=13,
                        type=int, help='dimensionality of the input feats')
    parser.add_argument('-utt-out-dim', default=3337,
                        type=int, help='dimensionality of the output')
    parser.add_argument('-utt-pad-lab', default=3336,
                        type=int, help='we use thid lable to do the padding')
    parser.add_argument('-layer-num', default=3,
                        type=int, help='number of conv layers')
    parser.add_argument('-kernel-size', default=4,
                        type=int, help='size of the conv kernels')
    parser.add_argument('-kernel-num', default=1024,
                        type=int, help='number of the conv filters')
    parser.add_argument('-last-kernel-size', default=1,
                        type=int, help='size of the conv kernels')
    parser.add_argument('-last-kernel-num', default=2048,
                        type=int, help='number of the conv filters')
    parser.add_argument('-l2', default=0.0,
                        type=float, help='l2 regularization factor')
    parser.add_argument('-activation', default='relu',
                        type=str, help='activation type')
    parser.add_argument('-batch-size', default=16,
                        type=int, help='size of the batch')
    parser.add_argument('-epochs-num', default=20,
                        type=int, help='number of training epochs')
    parser.add_argument('-step-size', default=0.001,
                        type=float, help='step size')
    prm = parser.parse_args()
    return prm


class Data:
    class Subset(Sequence):
        def __init__(self):
            self.ids = None
            self.X = None  # inputs
            self.y = None  # outputs
            self._l = None  # lengths
            # Following properties are for data generation
            self.shuffle = None
            self.gen_indexes = None

        def load(self, file_path, shuffle):
            self.shuffle = shuffle
            data_dic = pickle.load(open(file_path, 'rb'))
            self.ids = list(data_dic['input'].keys())
            self.X = []
            self.y = []
            self._l = []
            for id in self.ids:
                X = data_dic['input'][id]
                y = data_dic['output'][id]
                _l = X.shape[0]
                self.X.append(X)
                self.y.append(y)
                self._l.append(_l)
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self._l = np.array(self._l)
            self.ids = np.array(self.ids)
            self.sort_data()
            self.chunk_data()
            self.zero_pad_data()
            self.on_epoch_end()

        def sort_data(self):
            sort_inds = np.argsort(self._l)
            self.X = self.X[sort_inds]
            self.y = self.y[sort_inds]
            self._l = self._l[sort_inds]
            self.ids = self.ids[sort_inds]

        def zero_pad_data(self):
            for c in range(len(self.ids)):
                utt_len = self._l[c][-1]
                utt_num = len(self.ids[c])
                batch_X = np.zeros((utt_num, utt_len, prm.utt_in_dim))
                batch_y = prm.utt_pad_lab * np.ones((utt_num, utt_len), dtype=int)
                for u in range(utt_num):
                    batch_X[u, :self._l[c][u], :] = self.X[c][u]
                    batch_y[u, :self._l[c][u]] = self.y[c][u]
                self.X[c] = batch_X
                self.y[c] = batch_y

        def chunk_data(self):
            num_of_chunks = int(np.ceil(len(self.ids) / prm.batch_size))
            self.ids = np.array([self.ids[c * prm.batch_size: (c + 1) * prm.batch_size]
                                 for c in range(num_of_chunks)])
            self.X = np.array([self.X[c * prm.batch_size: (c + 1) * prm.batch_size]
                               for c in range(num_of_chunks)])
            self.y = np.array([self.y[c * prm.batch_size: (c + 1) * prm.batch_size]
                               for c in range(num_of_chunks)])
            self._l = np.array([self._l[c * prm.batch_size: (c + 1) * prm.batch_size]
                                for c in range(num_of_chunks)])

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.gen_indexes = np.arange(len(self.ids))
            if self.shuffle:
                np.random.shuffle(self.gen_indexes)

        def __len__(self):
            'Denotes the number of batches per epoch'
            return len(self.ids)

        def __getitem__(self, index):
            'Generate one batch of data'
            batch_y = np_utils.to_categorical(self.y[self.gen_indexes[index]], prm.utt_out_dim)
            return self.X[self.gen_indexes[index]], batch_y

    def __init__(self):
        self.tr = Data.Subset()
        self.de = Data.Subset()
        self.te = Data.Subset()

    def load(self):
        self.tr.load(prm.tr_file, shuffle=True)
        self.de.load(prm.de_file, shuffle=False)
        self.te.load(prm.te_file, shuffle=False)


class Net:
    def __init__(self):
        self._net = None

    def construct(self):
        inp = Input(shape=(None, prm.utt_in_dim))
        x = inp
        for i in range(prm.layer_num):
            activation = prm.activation
            kernel_num = prm.kernel_num
            kernel_size = prm.kernel_size
            if i == (prm.layer_num - 1):
                kernel_num = prm.last_kernel_num
                kernel_size = prm.last_kernel_size
                activation = prm.activation
            x = Conv1D(filters=kernel_num,
                       kernel_size=kernel_size,
                       strides=1,
                       padding='same',
                       data_format='channels_last',
                       dilation_rate=1,
                       activation=activation,
                       use_bias=True,
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros',
                       kernel_regularizer=l2(prm.l2),
                       bias_regularizer=l2(prm.l2),
                       activity_regularizer=None)(x)
        x = Conv1D(filters=prm.utt_out_dim,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   data_format='channels_last',
                   dilation_rate=1,
                   activation='softmax',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2(prm.l2),
                   bias_regularizer=l2(prm.l2),
                   activity_regularizer=None)(x)
        outp = x
        self._net = Model(inputs=inp, outputs=outp)
        self._net.compile(optimizer=Adam(lr=prm.step_size),
                          loss='categorical_crossentropy')

    def summary(self):
        cnf_dic = self._net.get_config()
        return json.dumps(cnf_dic, indent=4)

    def train_one_epoch(self, training_generator):
        self._net.fit_generator(
            generator=training_generator,
            epochs=1,
            steps_per_epoch=len(training_generator.ids))

    def predict(self, data_generator):
        y_true_all = []
        y_pred_all = []
        for (X, y_true) in data_generator:
            y_pred = self._net.predict(X)
            y_pred = np.argmax(y_pred, -1)
            y_true = np.argmax(y_true, -1)
            for u in range(y_true.shape[0]):
                this_y_pred = y_pred[u]
                this_y_true = y_true[u]
                this_y_pred = this_y_pred[this_y_true != prm.utt_pad_lab]
                this_y_true = this_y_true[this_y_true != prm.utt_pad_lab]
                y_pred_all.append(this_y_pred)
                y_true_all.append(this_y_true)
        return y_true_all, y_pred_all

    def predict_and_save_one_hot(self, data_generator, set_path):
        if data_generator.shuffle:
            print('Error:')
            print('data_generator cannot be shuffled when predict_and_save_one_hot is called.')
            exit()
        y_true_all = []
        y_pred_all = []
        for c in range(len(data_generator.ids)):
            for i in range(len(data_generator.ids[c])):
                y_pred = self._net.predict(data_generator.X[c][i:i+1, :, :])
                y_pred = y_pred[0, :, :]
                y_pred_comp = np.argmax(y_pred, -1)
                y_true_comp = data_generator.y[c][i, :]
                y_pred = y_pred[y_true_comp != prm.utt_pad_lab, :]
                y_pred_comp = y_pred_comp[y_pred_comp != prm.utt_pad_lab]
                y_true_comp = y_true_comp[y_true_comp != prm.utt_pad_lab]
                y_pred_all.append(y_pred_comp)
                y_true_all.append(y_true_comp)
                prob_pred_dir = prm.decode_out_dir + '/prob_outputs/' + set_path + '/' + data_generator.ids[c][i] + '/'
                if not os.path.exists(prob_pred_dir):
                    os.makedirs(prob_pred_dir)
                np.save(prob_pred_dir + 'y_pred.npy', y_pred)
                np.save(prob_pred_dir + 'y_pred_comp.npy', y_pred_comp)
                np.save(prob_pred_dir + 'y_true_comp.npy', y_true_comp)
        return y_true_all, y_pred_all

    def save(self, path):
        full_path = os.path.abspath(path)
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self._net.save(full_path)

    def load(self, path):
        self._net = load_model(path)


class Metric:
    @staticmethod
    def calc_acc(y_true, y_pred):
        S = 0.0
        N = 0.0
        for utt_i in range(len(y_true)):
            S += np.sum(np.equal(y_true[utt_i], y_pred[utt_i]))
            N += len(y_true[utt_i])
        return S / N


def write_log(text):
    if not os.path.exists(prm.out_dir):
        os.makedirs(prm.out_dir)
    print(text)
    with open(prm.out_dir + '/log.txt', 'a') as myfile:
        myfile.write(text + '\n')


def save_results(y_true, y_pred, ids, path):
    full_path = os.path.abspath(path)
    dir_path = os.path.dirname(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'w') as handle:
        for i in range(len(ids)):
            handle.write('id: ' + ids[i] + '\n\n')
            handle.write('    y_true: ' + ', '.join(y_true[i].astype('str')) + '\n\n')
            handle.write('    y_pred: ' + ', '.join(y_pred[i].astype('str')) + '\n\n')
            handle.write('----------------\n\n')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
prm = parse_arguments()
write_log('Parameters: \n' + json.dumps(vars(prm), indent=4))
write_log('Loading data ...')
db = Data()
db.load()
write_log('Defining the network ...')
net = Net()
if prm.decode:
    write_log('Loading the network for decode ...')
    net.load(prm.decode_net_file)
    write_log('Evaluating the network ...')
    y_true_de, y_pred_de = net.predict_and_save_one_hot(db.de, 'de')
    write_log('    acc on dev = ' + str(Metric.calc_acc(y_true_de, y_pred_de)))
    y_true_te, y_pred_te = net.predict_and_save_one_hot(db.te, 'te')
    write_log('    acc on te = ' + str(Metric.calc_acc(y_true_te, y_pred_te)))
    exit()
write_log('Constructing the network ...')
net.construct()
write_log('Network config: \n' + net.summary())
write_log('Training the network ...')
write_log('Date-time: ' + str(datetime.datetime.now()))

for e in range(prm.epochs_num):
    write_log('Epoch ' + str(e))
    net.train_one_epoch(db.tr)
    write_log('Evaluating the network ...')
    y_true_de, y_pred_de = net.predict(db.de)
    write_log('    acc on dev = ' + str(Metric.calc_acc(y_true_de, y_pred_de)))
    y_true_te, y_pred_te = net.predict(db.te)
    write_log('    acc on te = ' + str(Metric.calc_acc(y_true_te, y_pred_te)))
    net.save(prm.out_dir + '/trained_models/iter' + str(e) + '.h5')
    write_log('Date-time: ' + str(datetime.datetime.now()))

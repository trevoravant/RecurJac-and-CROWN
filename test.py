import sys
import random

import numpy as np
import tensorflow as tf
#from keras import backend as K
from types import SimpleNamespace
from tensorflow.keras import backend as K

from setup_mnist import MNIST
from mnist_cifar_models import NLayerModel, get_model_meta
from utils import generate_data
from bound_base import get_weights_list
import compnet as exp

#orig = True
orig = False

class NLayerModel:
    def __init__(self, params, restore = None, session=None, use_softmax=False, image_size=28, image_channel=1, activation='relu', activation_param = 0.3, l2_reg = 0.0, dropout_rate = 0.0):
        
        global Sequential, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, LeakyReLU, regularizers, K
        if 'Sequential' not in globals():
            print('importing Keras from tensorflow...')
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
            from tensorflow.keras.layers import Conv2D, MaxPooling2D
            from tensorflow.keras.layers import LeakyReLU
            from tensorflow.keras.models import load_model
            from tensorflow.keras import regularizers
            from tensorflow.keras import backend as K
        
        self.image_size = image_size
        self.num_channels = image_channel
        self.num_labels = 10
        
        model = Sequential()
        #model.add(Flatten(input_shape=(image_size, image_size, image_channel)))
        if orig:
            model.add(Flatten(input_shape=(28, 28, 1)))
        else:
            model.add(Flatten(input_shape=(exp.n_input, 1)))
        #model.add(Flatten())
        # list of all hidden units weights
        self.U = []
        n = 0
        for param in params:
            n += 1
            # add each dense layer, and save a reference to list U
            self.U.append(Dense(param, kernel_initializer = 'he_uniform', kernel_regularizer=regularizers.l2(l2_reg)))
            model.add(self.U[-1])
            # ReLU activation
            # model.add(Activation(activation))
            if activation == "arctan":
                model.add(Lambda(lambda x: tf.atan(x), name=activation+"_"+str(n)))
            elif activation == "leaky":
                print("Leaky ReLU slope: {:.3f}".format(activation_param))
                model.add(LeakyReLU(alpha = activation_param, name=activation+"_"+str(n)))
            else:
                model.add(Activation(activation, name=activation+"_"+str(n)))
            if dropout_rate > 0.0:
                model.add(Dropout(dropout_rate))
        if orig:
            n_output = 10
        else:
            n_output = exp.n_output
        self.W = Dense(n_output, kernel_initializer = 'he_uniform', kernel_regularizer=regularizers.l2(l2_reg))
        model.add(self.W)
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        # save the output of intermediate layers
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        # a tensor to get gradients
        self.gradients = []
        for i in range(model.output.shape[1]):
            output_tensor = model.output[:,i]
            self.gradients.append(K.gradients(output_tensor, model.input)[0])

        self.layer_outputs = layer_outputs
        self.model = model
        model.summary()

    def predict(self, data):
        return self.model(data)
    
    def get_gradient(self, data, sess = None):
        if sess is None:
            sess = K.get_session()
        # initialize all un initialized variables
        # sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))]))
        evaluated_gradients = []
        for g in self.gradients:
            evaluated_gradients.append(sess.run(g, feed_dict={self.model.input:data}))
        return evaluated_gradients

args = SimpleNamespace()
args.seed = 1228
args.numimage = 1
args.startimage = 0
args.task = 'lipschitz'
args.jacbndalg = 'recurjac'
args.liplogstart = -3
args.liplogend = 0.0
args.lipsteps = 20
args.eps = 0.005
args.layerbndalg = 'crown-adaptive'
args.norm = float("inf")
#args.norm = 2
args.quad = False
args.lipsshift = 1
args.lipsdir = -1

if orig:
    numlayer = 7
else:
    numlayer = exp.numlayer

targeted = True
force_label = 1
target_type = None
activation_param = None
activation = 'relu'
if orig:
    modelfile = '/home/trevor/RecurJac-and-CROWN/models/mnist_7layer_relu_1024'
else:
    modelfile = 'tf_model.h5'
weight_dims, activation, activation_param, input_dim = get_model_meta(modelfile)

config = tf.compat.v1.ConfigProto() #TDA
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config) as sess: #TDA
    data = MNIST()
    model = NLayerModel(weight_dims[:-1], modelfile, activation=activation, activation_param=activation_param)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed) #TDA

    weights, biases = get_weights_list(model)

    if orig:
        inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=data.test_labels.shape[0], total_images = args.numimage, targeted=targeted, random_and_least_likely = True, force_label = force_label, target_type = target_type, predictor=model.model.predict, start=args.startimage)
        preds = model.model.predict(inputs)
    else:
        inputs = np.zeros([1,exp.n_input,1])
        targets = np.zeros([1,exp.n_output])
        true_labels = np.zeros([1,exp.n_output])
        true_ids = np.array([2]) # I don't think this matters in this case
        img_info = ['1'] # I don't think this matters in this case
        preds = model.model.predict(inputs,steps=1)

    task_input = locals()
    task_modudle = __import__("task_"+args.task)
    task = task_modudle.task(**task_input)
    task.run_single(0)
    sys.stdout.flush()
    sys.stderr.flush()
task.summary()
sys.stdout.flush()
sys.stderr.flush()
#print(lipschitz_const)
'''
bounded_input=False
wramup=False
targettype='1'
steps=15
seed=1228
'''

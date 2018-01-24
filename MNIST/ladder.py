import tensorflow as tf
import input_data
import math
import os
import csv
from functools import reduce
import numpy as np
import sys
import time
import pandas as pd
from utils import flat, max_pool_2x2, unpool, run_layer, run_transpose_layer


#layers = [{"type": "conv2d", "kernel_shape": [5,5,1,32]}, # 28x28x1 -> 28x28x32
#           {"type": "max_pool_2x2"},                      # 28x28x32 -> 14x14x32
#           {"type": "conv2d", "kernel_shape": [5,5,32,64]},# 14x14x32 -> 14x14x64
#           {"type": "max_pool_2x2"},                       # 14x14x64 -> 7x7x64
#           {"type": "flat"},                               # 7x7x64 -> 3136
#           {"type": "dense", "kernel_shape": [7*7*64, 1024]},# 3136 -> 1024
#           {"type": "dense", "kernel_shape": [1024, 10]}]    # 1024 -> 10


layers = [ {"type": "dense", "kernel_shape": [784, 1000]},
           {"type": "dense", "kernel_shape": [1000, 500]},
           {"type": "dense", "kernel_shape": [500, 250]},
           {"type": "dense", "kernel_shape": [250, 250]},
           {"type": "dense", "kernel_shape": [250, 250]},
           {"type": "dense", "kernel_shape": [250, 10]}]

# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

checkpoint_dir = "checkpoints/"

image_shape = [784]

num_epochs = 150
starter_learning_rate = 0.02
decay_after = 15  # epoch after which to begin learning rate decay
batch_size = 100
num_labeled = 100

noise_std = 0.3  # scaling factor for noise used in corrupted encoder

#==================================================================================================

L = len(layers)  # number of layers

tf.reset_default_graph()

# feedforward_inputs (FFI): inputs for the feedforward network (i.e. the encoder).
# Should contain the labeled images during train time and test images during test time.
feedforward_inputs = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name = "FFI")

# autoencoder_inputs (AEI): inputs for the autoencoder (encoder + decoder).
# Should contain the unlabeled images during train time.
autoencoder_inputs = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name = "AEI")

outputs = tf.placeholder(tf.float32)# labels for the labeled images
training = tf.placeholder(tf.bool) # If training or not

FFI = tf.reshape(feedforward_inputs, [-1] + image_shape)
AEI = tf.reshape(autoencoder_inputs, [-1] + image_shape)

#==================================================================================================
# Batch normalization functions
ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def update_batch_normalization(batch, output_name = "bn", scope_name = "BN"):
    dim = len(batch.get_shape().as_list())
    mean, var = tf.nn.moments(batch, axes=list(range(0,dim-1)))
    # Function to be used during the learning phase. Normalize the batch and update running mean and variance.
    with tf.variable_scope(scope_name, reuse= tf.AUTO_REUSE):
        running_mean = tf.get_variable("running_mean", mean.shape, initializer= tf.constant_initializer(0))
        running_var = tf.get_variable("running_var", mean.shape, initializer= tf.constant_initializer(1))
    
    assign_mean = running_mean.assign(mean)
    assign_var = running_var.assign(var)
    bn_assigns.append(ewma.apply([running_mean, running_var]))
    
    with tf.control_dependencies([assign_mean, assign_var]):
        z = (batch - mean) / tf.sqrt(var + 1e-10)
        return tf.identity(z, name = output_name)


def batch_normalization(batch, mean=None, var=None, output_name = "bn"):
    dim = len(batch.get_shape().as_list())
    mean, var = tf.nn.moments(batch, axes=list(range(0,dim-1)))
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    z = (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
    return tf.identity(z, name = output_name)

#______________________________________________________________________________________
# Encoder
    
def encoder_bloc(h, layer_spec, noise_std, update_BN, activation):
    # Run the layer
    z_pre = run_layer(h, layer_spec, output_name="z_pre")
    
    # Compute mean and variance of z_pre (to be used in the decoder)
    dim = len(z_pre.get_shape().as_list())
    mean, var = tf.nn.moments(z_pre, axes=list(range(0,dim-1)))
    mean = tf.identity(mean, name = "mean")
    var = tf.identity(var, name = "var")
    
    # Batch normalization
    def training_batch_norm():
        if update_BN:
            z = update_batch_normalization(z_pre)
        else:
            z = batch_normalization(z_pre)
        return z

    def eval_batch_norm():
        with tf.variable_scope("BN", reuse= tf.AUTO_REUSE):
            mean = ewma.average(tf.get_variable("running_mean", shape = z_pre.shape[-1]))
            var = ewma.average(tf.get_variable("running_var", shape = z_pre.shape[-1]))
        z = batch_normalization(z_pre, mean, var)
        return z
    
    # Perform batch norm depending to the phase (training or testing)
    z = tf.cond(training, training_batch_norm, eval_batch_norm)
    z += tf.random_normal(tf.shape(z)) * noise_std
    z = tf.identity(z, name = "z")
    
    # Center and scale plus activation
    size = z.get_shape().as_list()[-1]
    beta = tf.get_variable("beta", [size], initializer= tf.constant_initializer(0))
    gamma = tf.get_variable("gamma", [size], initializer= tf.constant_initializer(1))
    
    h = activation(z*gamma + beta)
    return tf.identity(h, name = "h")


def encoder(h, noise_std, update_BN):
    # Perform encoding for each layer
    h += tf.random_normal(tf.shape(h)) * noise_std
    h = tf.identity(h, "h0")
    
    for i, layer_spec in enumerate(layers):
        with tf.variable_scope("encoder_bloc_" + str(i+1), reuse = tf.AUTO_REUSE):
            # Create an encoder bloc if the layer type is dense or conv2d
            if layer_spec["type"] == "flat":
                h = flat(h, output_name="h")
            elif layer_spec["type"] == "max_pool_2x2":
                h = max_pool_2x2(h, output_name="h")
            else:
                if i == L-1:
                    activation = tf.nn.softmax # Only for the last layer
                else:
                    activation = tf.nn.relu
                h = encoder_bloc(h, layer_spec, noise_std, update_BN = update_BN, activation = activation)
                
    y = tf.identity(h, name = "y")
    return y
    
#__________________________________________________________________________________________________________
# Decoder

def g_gauss(z_c, u, output_name="z_est", scope_name = "denoising_func"):
#    gaussian denoising function proposed in the original paper
    size = u.get_shape().as_list()[-1]
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE):
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')
    
        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
    return tf.identity(z_est, name = output_name)


def decoder_bloc(u, z_corr, mean, var, layer_spec=None):
    # Performs the decoding operations of a corresponding encoder bloc
    # Denoising
    z_est = g_gauss(z_corr, u)
    
    z_est_BN = (z_est - mean)/tf.sqrt(var + tf.constant(1e-10))
    z_est_BN = tf.identity(z_est_BN, name = "z_est_BN")
    
    # run transposed layer
    if layer_spec is not None:
        u = run_transpose_layer(z_est, layer_spec)
        u = batch_normalization(u, output_name = "u")
    
    return u, z_est_BN

# ========================================================================================================
# Graph building
print("===  Building graph ===")

# Encoder
with tf.name_scope("FF_clean"):
    FF_y = encoder(FFI, 0, update_BN = False) # output of the clean encoder. Used for prediction
with tf.name_scope("FF_corrupted"):
    FF_y_corr = encoder(FFI, noise_std, update_BN = False) # output of the corrupted encoder. Used for training.

with tf.name_scope("AE_clean"):
    AE_y = encoder(AEI, 0, update_BN = True) # Clean encoding of unlabeled images
with tf.name_scope("AE_corrupted"):
    AE_y_corr = encoder(AEI, noise_std, update_BN = False) # corrupted encoding of unlabeled images
    

#__________________________________________________________________________________________________________
# Decoder
    # Function used to get a tensor from encoder
get_tensor = lambda input_name, num_encoder_bloc, name_tensor: tf.get_default_graph().get_tensor_by_name(
            input_name + "/encoder_bloc_" + str(num_encoder_bloc) + "/" + name_tensor + ":0")


d_cost = []
u = batch_normalization(AE_y_corr, output_name="u_L") 
for i in range(L,0,-1):
    layer_spec = layers[i-1]
    
    with tf.variable_scope("decoder_bloc_" + str(i), reuse = tf.AUTO_REUSE):
        # if the layer is max pooling or "flat", the transposed layer is run without creating a decoder bloc.
        if layer_spec["type"] in ["max_pool_2x2", "flat"]:
            h = get_tensor("AE_corrupted",i-1,"h")
            output_shape = tf.shape(h)
            u = run_transpose_layer(u, layer_spec, output_shape=output_shape)
        else:
            z_corr, z = [get_tensor("AE_corrupted",i,"z"), get_tensor("AE_clean",i,"z")]
            mean, var = [get_tensor("AE_clean",i,"mean"), get_tensor("AE_clean",i,"var")]
                
            u, z_est_BN = decoder_bloc(u, z_corr, mean, var, layer_spec=layer_spec)
            d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[i])


# last decoding step
with tf.variable_scope("decoder_bloc_0", reuse = tf.AUTO_REUSE):
    z_corr = tf.get_default_graph().get_tensor_by_name("AE_corrupted/h0:0")
    z = tf.get_default_graph().get_tensor_by_name("AE_clean/h0:0")
    mean, var = tf.constant(0.0), tf.constant(1.0)
    
    z_est_BN = decoder_bloc(u, z_corr, mean, var)
    d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[0])


# Loss and accuracy
u_cost = tf.add_n(d_cost) # decoding cost
corr_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(FF_y_corr), 1))  # supervised cost
clean_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(FF_y), 1))  # cost used for prediction

loss = corr_pred_cost + u_cost  # total cost

correct_prediction = tf.equal(tf.argmax(FF_y, 1), tf.argmax(outputs, 1))  # number of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)


# Optimization setting
learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)


n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print(str(n) + " trainable parameters")


#========================================================================================================================
    # Learning phase
print("===  Loading Data ===")
data = input_data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)
num_examples = data.train.unlabeled_ds.images.shape[0]
num_iter = (num_examples//batch_size) * num_epochs  # number of loop iterations

print("===  Starting Session ===")
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print("=== Training ===")
initial_accuracy = sess.run(accuracy, feed_dict={
                                      feedforward_inputs: data.test.images, 
                                      outputs: data.test.labels, 
                                      training: False})
print("Initial Accuracy: ", initial_accuracy, "%")

i_iter = 0
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
#    i_iter = (epoch_n+1) * (num_examples/batch_size)
    i_iter = (epoch_n+1) * (num_examples//batch_size) #py3 adapt
    print("Restored Epoch ", epoch_n)
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    init = tf.global_variables_initializer()
    sess.run(init)

for i in range(i_iter, num_iter):
    labeled_images, labels, unlabeled_images = data.train.next_batch(batch_size)
    
    T = time.time()
    sess.run(train_step, feed_dict={feedforward_inputs: labeled_images, 
                                    outputs: labels, 
                                    autoencoder_inputs: unlabeled_images, 
                                    training: True})
    T = time.time() - T
    
#    msg = "\rIterations : {:} step time : {:}".format(i,T)
#    sys.stdout.write(msg)
#    sys.stdout.flush()
    
    if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
        # Compute train loss, train accuracy and test accuracy for each epoch
        epoch_n = i//(num_examples//batch_size)
        
        train_loss = sess.run(loss, feed_dict={feedforward_inputs: labeled_images, 
                                               outputs: labels, 
                                               autoencoder_inputs: unlabeled_images, 
                                               training: False})
    
        train_acc  = sess.run(accuracy, feed_dict={feedforward_inputs: data.train.labeled_ds.images, 
                                                   outputs: data.train.labeled_ds.labels, 
                                                   training: False})
    
        test_acc = sess.run(accuracy, feed_dict={feedforward_inputs: data.test.images, 
                                                 outputs: data.test.labels, 
                                                 training: False})
        
        print("\nEpoch ", str(int(epoch_n+0.1)), ": train loss = {:.5f}, train accuracy = {:.3}, test accuracy = {:.3}".format(train_loss,train_acc,test_acc))
        
        if (epoch_n+1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
            
        saver.save(sess, checkpoint_dir + 'model.ckpt', epoch_n)


final_accuracy = sess.run(accuracy, feed_dict={feedforward_inputs: data.test.images, 
                                          outputs: data.test.labels, 
                                          training: False})
print("Final Accuracy: ", final_accuracy, "%")
    
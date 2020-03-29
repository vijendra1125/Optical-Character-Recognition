'''
@Author: Vijendra Singh
'''

import os
from datetime import datetime as dt
import string
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def read_data(file_paths):
  '''
  @brief: read data from tfrecords file
  @args[in]:
    file_paths: list of path to tfrecord files
  @args[out]:
    image: an image being read from tfrecord
    label: a label being read from tfrecord corresponding to image
  '''
  file_queue=tf.train.string_input_producer(file_paths)
  feature = {'images': tf.FixedLenFeature([], tf.string),
             'labels': tf.FixedLenFeature([], tf.string)}    
  reader = tf.TFRecordReader()  
  _,record=reader.read(file_queue)
  features = tf.parse_single_example(record, features=feature)
  image = tf.decode_raw(features['images'], tf.uint8)
  label = tf.decode_raw(features['labels'], tf.uint8) 
  return image,label


def minibatch(batch_size, 
              file_paths, 
              image_size, 
              string_length, 
              class_count):
  '''
  @brief: create minibatch of data (iamge and label)
  @args[in]:
    batch_size: size of the minibatch
    file_paths: list of path to the files
    image_size: size of the image (row, columns, channels)
    string_length: length of label string (including whitespace)
    class_count: total number of classes
  @args[out]:
    image_batch: batch of image
    label_batch: batch of label
  ''' 
  image, label=read_data(file_paths)
  image = tf.cast(tf.reshape(image,image_size), dtype = tf.float32)
  label = tf.reshape(label, [1, string_length])
  label = tf.one_hot(label, class_count,axis=1)
  label = tf.reshape(label, tf.shape(label)[1:])
  image_batch,label_batch= tf.train.shuffle_batch([image, label],
                          batch_size, capacity, min_after_dequeue,
                          num_threads = num_of_threads)
  label_batch = tf.cast(label_batch, dtype = tf.int64)
  return image_batch, label_batch


def variable(name, shape, initializer, weight_decay = None):
  '''
  @brief: create parameter tensor
  '''
  var = tf.get_variable(name, shape, initializer = initializer)
  if weight_decay is not None:
    weight_loss=tf.multiply(tf.nn.l2_loss(var),weight_decay,name="weight_loss")
    tf.add_to_collection('losses', weight_loss)
  return var


def conv_block(block_num,
               input_data,
               weights, 
               weight_initializer=tf.contrib.layers.xavier_initializer(),
               bias_initializer=tf.constant_initializer(0.0),
               conv_op=[1,1,1,1],
               conv_padding='SAME',
               weight_decay=None,
               lrn=True,
               dropout=1.0, 
               activation=True):
  '''
  @brief: convolutional block
  '''
  with tf.variable_scope('conv'+ str(block_num), reuse = tf.AUTO_REUSE) as scope:
    input_data = tf.nn.dropout(input_data, dropout)
    kernel = variable('weights', weights, initializer = weight_initializer, weight_decay = weight_decay)
    biases = variable('biases', weights[3], initializer=bias_initializer, weight_decay=None)
    conv = tf.nn.conv2d(input_data, kernel, conv_op, padding=conv_padding)
    pre_activation = tf.nn.bias_add(conv, biases)
    if lrn==True:
      pre_activation = tf.nn.lrn(pre_activation, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm')
    if activation:
      conv_out = tf.nn.relu(pre_activation, name=scope.name)
      return conv_out
    else:
      return pre_activation


def dense_block(block_num,
                input_data,
                neurons,
                weight_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.0),
                weight_decay=None,
                activation=True, 
                dropout=1.0):
  '''
  @brief: Fully connected block
  '''
  with tf.variable_scope('dense'+ str(block_num), reuse = tf.AUTO_REUSE) as scope:
    input_data = tf.nn.dropout(input_data, dropout)
    weights = variable('weights', [input_data.shape[1], neurons], initializer=weight_initializer, weight_decay = weight_decay)
    biases = variable('biases', [1,neurons], initializer = bias_initializer, weight_decay = None)
    dense = tf.matmul(input_data,weights)+biases
    if activation:
      dense=tf.nn.relu(dense, name=scope.name)
    return dense
  
  
def multi_loss(logits, labels, batch_size, max_char):
  '''
  @brief: cross entopy loss for multi class
  '''
  loss = 0
  for i in range(max_char):
    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
            (logits=logits[:,i,:],labels=labels[:,:,i]), \
                           name='cross_entropy_loss_mean')
  loss /= max_char
  tf.add_to_collection('losses', loss)
  total_loss=tf.add_n(tf.get_collection('losses'), name='total_loss')
  tf.add_to_collection('losses', total_loss)
  return total_loss


def parameter_update(loss, learning_rate):
  '''
  @brief: optimization and parameter update using adam optimizer
  '''
  optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  return optimizer


def accuracy_calc(output, label_batch):
  '''
  @brief: calculate accuracy
  '''
  correct_prediction = tf.equal(tf.cast(tf.argmax(output, 2),dtype=tf.int32),tf.cast(tf.argmax(label_batch, 1),dtype=tf.int32))
  accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
  return accuracy

  def inference(image_batch, class_count,
              dropout=[1,1,1,1],
              wd=None):
  '''
  @brief: define architecture using building block fuctions above
  '''
  i = 0
  weights=[[3,3,1,class_count//4],
           [3,3,class_count//4,class_count//2],
           [3,3,class_count//2,class_count],
           [3,3,class_count,class_count]]
  conv_op=[[1,1,1,1],[1,1,1,1],[1,1,1,1], [1,1,1,1]]
  
  conv1 = conv_block(1,image_batch,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
  i=i+1
  pool1=tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1,2,2,1],padding='VALID', name='pool1') #16x128
  
  conv2 = conv_block(2,pool1,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
  i=i+1
  pool2=tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1,2,2,1],padding='VALID', name='pool2') #8x64
  
  conv3 = conv_block(3,pool2,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
  i=i+1
  pool3=tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1,2,2,1],padding='VALID', name='pool3') #4x32
  
  conv4 = conv_block(4,pool3,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
  pool4=tf.nn.max_pool(conv4, ksize=[1, 4, 2, 1], strides=[1,1,2,1],padding='VALID', name='pool4') #1x16
  
  flat=tf.reshape(pool4, [tf.shape(image_batch)[0], string_length, class_count], name='flat')
		
  return flat

   def decoding(encoded_data, type = 'logit'):
  '''
  @brief: decoding
  '''
  if(type == 'logit'):
    prediction = np.argmax(encoded_data, 2)
  elif(type == 'label'):
    prediction = np.argmax(encoded_data, 1)
  decoded_prediction = []
  for dp in prediction:
    predicted_text = ''
    for p in dp:
      predicted_text += all_chr[p]
    decoded_prediction.append(predicted_text)
  return decoded_prediction


def eval_vizualization(X):
  '''
  @brief: visualization of evaluation result
  '''
  decoded_text = []
  logit = inference(X, class_count)
  init=tf.global_variables_initializer()
  saver=tf.train.Saver()
  
  with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,checkpoint_restore)
    text = sess.run(logit)
    decoded_text = decoding(text, type = 'logit')
  for i in range(X.shape[0]):
    x = np.reshape(X[i, :,:,:], image_size[0:2])
    plt.imshow(x, cmap = 'gray')
    plt.show()
    print("text: ", decoded_text[i], '<---')
'''
@Author: Vijendra Singh
'''
import string
from datetime import datetime as dt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def _bytes_feature(value):
  '''
  @brief: function supporting write_tfrecord function below
  ''' 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecords(images, labels, file_path):
  '''
  @brief: write data to a tfrecords file
  args[in]:
    images: all image data
    label: all label data
    file_path: path of tfrecord file in which data need to be written
  '''
  start_time=dt.now()
  writer = tf.python_io.TFRecordWriter(file_path)
  for image, label in zip(images, labels):
      feature = {'labels': _bytes_feature(tf.compat.as_bytes(np.array(label).tostring())),
                 'images': _bytes_feature(tf.compat.as_bytes(np.array(image).tostring()))}
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())    
  writer.close()
  end_time = dt.now()  
  print("time taken to write data", end_time - start_time)

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


def minibatch(file_paths, 
              image_size, 
              string_length, 
              class_count,
              batch_size,
              capacity,
              min_after_dequeue, 
              number_of_threads):
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
                          number_of_threads)
  label_batch = tf.cast(label_batch, dtype = tf.int64)
  return image_batch, label_batch

def gen_rand_string_data(data_count,                        
                         min_char_count = 3, 
                         max_char_count = 8,
                         string_length = 16,
                         x_pos = 'side',
                         image_size = (32,256,1),
                         font = [cv2.FONT_HERSHEY_SIMPLEX], 
                         font_scale = np.arange(0.7, 1, 0.1), 
                         thickness = range(1, 3, 1)):
  '''
  @brief: random string data generation
  @args[in]:
    data_count: number of data sample to be genrated
    min_char_count: minimum number of character in a string (exclusing whitespace)
    max_char_count: maximum number of character in a string (excluding whitespace)
    string_length: maximum of number of character in string (including whitespace)
    x_pose: where to position string in image ("side" for side of image, anything else for center of image)
    image_size: size of data image
    font: lis to fonts (choosen from what available in opencv)
    font_scale: list of font scale/size
    thickness: list of font thickness
  @args[out]:
    images: all generated images
    labels: labels for all generated images
  ''' 
  start_time=dt.now() 
  images = []
  labels = []
  #set text color to white
  color = (255,255,255) 
  # prepare the list of characters to consider
  char_list = list(string.ascii_letters) \
              + list(string.digits) \
              + list(' ')     
  for count in range(data_count):  
    for fs in font_scale:
      for thick in thickness:
        for f in font:
          # generate image with black background
          img = np.zeros(image_size, np.uint8)
          # generate random string within given constraint
          char_count = np.random.randint(min_char_count, (max_char_count + 1))
          rand_str = ''.join(np.random.choice(char_list, char_count))
          # generate image data
          text_size = cv2.getTextSize(rand_str, f, fs, thick)[0]  
          if(x_pos == 'side'):
            org_x = 0
          else:
            org_x = (image_size[1] - text_size[0])//2         
          org_y = (image_size[0] +  text_size[1])//2
          cv2.putText(img, rand_str, (org_x, org_y), f, fs, color, thick, cv2.LINE_AA)
          # prepare label
          label = list(rand_str) + [' '] * (string_length - len(rand_str))
          for i,t in enumerate(label):
            label[i] = char_list.index(t)           
          label = np.uint8(label)
          images.append(img)
          labels.append(label)        
  end_time = dt.now()  
  print("time taken to generate data", end_time - start_time)          
  return images, labels
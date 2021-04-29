from pathlib import Path
import shutil

from tqdm.notebook import tqdm
import collections
import random
import re
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tensorflow as tf
import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def download_cocodataset(annotation_path='/annotations/', image_path='/train2017/'):
  # Download caption annotation files
  annotation_folder = annotation_path
  if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                            extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'
    os.remove(annotation_zip)

  # Download image files
  image_folder = image_path
  if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train2017.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/train2017.zip',
                                        extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
  else:
    PATH = os.path.abspath('.') + image_folder

def caption_and_img_vector(train_image_paths, image_path_to_caption):
  train_captions = []
  img_name_vector = []

  for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]

    train_captions.extend(caption_list) 
    img_name_vector.extend([image_path] * len(caption_list)) 

  return train_captions, img_name_vector



def load_image(image_path):
  '''
  return (229,299,3), image_path
  img를 불러와서 inceptionV3에 넣기위해 
  최소 전처리(이미지 size변경, preprocess_input 수행)
  '''
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

def calc_max_length(tensor):
    return max(len(t) for t in tensor) 



def train_test_splits(img_name_vector, cap_vector, ratio=0.8):

  # train / test split
  img_to_cap_vector = collections.defaultdict(list)
  for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

  img_keys = list(img_to_cap_vector.keys())
  # keys를 셔플
  random.shuffle(img_keys) 

  #훈련셋 80% 테스트셋 20%
  slice_index = int(len(img_keys)*ratio) 
  img_name_train_keys, img_name_val_keys = img_keys[:slice_index] , img_keys[slice_index:]

  img_name_train = []
  cap_train = []
  for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    img_name_train.extend([imgt] * capt_len)
    cap_train.extend(img_to_cap_vector[imgt])

  img_name_val = []
  cap_val = []
  for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    img_name_val.extend([imgv] * capv_len)
    cap_val.extend(img_to_cap_vector[imgv])
  print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

  return img_name_train, cap_train, img_name_val, cap_val, img_to_cap_vector



def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


@tf.autograph.experimental.do_not_convert
def dataset_pipeline(img_name_train, cap_train, img_name_val, cap_val, BUFFER_SIZE, BATCH_SIZE):
  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

  # map_func을 불러서, 저장한 특징값 배열을 불러옴
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]), 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  #dataset의 첫번째 값은 img_tensor(특징값),두번째는 caption

  # 버퍼사이즈만큼 올려서 랜덤으로 섞고 배치만큼 나눠줌
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  # test set도 똑같이해줌
  v_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))


  v_dataset = v_dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # 섞고 배치하기
  v_dataset = v_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  v_dataset = v_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset, v_dataset


def learing_curve(loss_plot, v_loss_plot):
  plt.plot(loss_plot, label='train loss')
  plt.plot(v_loss_plot, label='val loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Plot')
  plt.legend()
  plt.show()



def blue_cos_score(blue_score_plot, cosin_score_plot):
  fig = plt.figure(figsize=(15,5))
  ax = plt.subplot(1,2,1)
  for i in range(4):
    # ax = plt.subplot(1,3,i)
    if i < 3:
      labels = f'{i+1}-gram score'
    else :
      labels = f'average-gram score'
    plt.plot(np.array(blue_score_plot).T[i], label=labels)
  plt.xlabel('Epochs')
  plt.ylabel('Score')
  plt.title('BLEU Score Plot')
  plt.legend()

  ax = plt.subplot(1,2,2)
  plt.plot(cosin_score_plot, label='Score')
  plt.xlabel('Epochs')
  plt.ylabel('Score')
  plt.title('Cosine Similarity')
  # plt.legend()
  plt.show()


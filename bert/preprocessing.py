import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.bert.configs
import official.nlp.bert.tokenization

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# tfds로부터 Dataset 읽어오는 함수
def download_data(name):
    datas, labels = tfds.load(name, with_info = True, batch_size =-1)
    return datas, labels

def return_tokenizer(path, vocabulary):
    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file = os.path.join(path, vocabulary), 
        do_lower_case = True ## 소문자로 고침
    )
    return tokenizer

# [SEP] 토큰 추가하는 함수
def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s.numpy()))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

# [CLS] 토큰 추가 및 BERT에 필요한 입력 요소들을 생성하는 함수
def bert_encode(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer) for s in glue_dict["sentence1"]])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer) for s in glue_dict["sentence2"]])
    
    # CLS 토큰 선언
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0] 
    
    # CLS 토큰을 통해 sentense1과 sentense2를 통합
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1) 
    
    input_mask = tf.ones_like(input_word_ids).to_tensor() 

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs
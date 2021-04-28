import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.run_classifier
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

from bert.preprocessing import *

epochs = 3
batch_size = 32
eval_batch_size = 32
dataName = 'glue/mrpc'
bert_dir = "./bertModel/"
bert_base = "multi_cased_L-12_H-768_A-12"
bert_small = "uncased_L-8_H-512_A-8"
bert_smaller = "uncased_L-4_H-512_A-8"
vocab = "vocab.txt"
bertConfig = 'bert_config.json'

gs_folder_bert = bert_dir+bert_base
#gs_folder_bert = bert_dir+bert_small
#gs_folder_bert = bert_dir+bert_smaller

# BERT 모델 정의 Config 파일 불러오기
def return_bert_config(path, fileName):
    bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
    bert_config = bert.configs.BertConfig.from_dict(config_dict)

    return config_dict, bert_config

def main():
    glue, info = download_data(dataName)

    tokenizer = return_tokenizer(gs_folder_bert, vocab)

    glue_train = bert_encode(glue['train'], tokenizer)
    glue_train_labels = glue['train']['label']

    glue_validation = bert_encode(glue['validation'], tokenizer)
    glue_validation_labels = glue['validation']['label']

    glue_test = bert_encode(glue['test'], tokenizer)
    glue_test_labels  = glue['test']['label']

    config_dict, bert_config = return_bert_config(gs_folder_bert, bertConfig)

    bert_classifier, bert_encoder = bert.bert_models.classifier_model(
        bert_config, num_labels=2)

    checkpoint = tf.train.Checkpoint(encoder = bert_encoder)

    train_data_size = len(glue_train_labels)
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch*epochs
    warmup_steps = int(epochs*train_data_size*0.1/batch_size)

    optimizer = nlp.optimization.create_optimizer(
        2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    bert_classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    bert_classifier.fit(
        glue_train, glue_train_labels,
        validation_data=(glue_validation, glue_validation_labels),
        batch_size=32,
        epochs = 10)

    export_dir = './saved_model'
    tf.saved_model.save(bert_classifier, export_dir=export_dir)

    print("Training_complete")

if __name__=="__main__":
    main()
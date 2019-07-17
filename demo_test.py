# %%
import os
import pandas as pd
import tensorflow as tf
import argparse
from lib.setup import params_setup, logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters
from lib.train import train
from lib.test import test
from lib.data_generator import customDataGenerator

# %%
time_series_df = pd.read_parquet('./data/raw_time_series.parquet')
time_series_df.sample()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--attention_len', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_set', type=str, default='sample_data')
parser.add_argument('--decay', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--file_output', type=int, default=1)
parser.add_argument('--highway', type=int, default=8)
parser.add_argument('--horizon', type=int, default=4)
parser.add_argument('--init_weight', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--max_gradient_norm', type=float, default=5.0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model_dir', type=str, default='./models/model')
parser.add_argument('--mts', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_units', type=int, default=338)
parser.add_argument('--custom',type=bool,default=True)
parser.add_argument('--split_date',type=list,default=['20181201','20190320'])
parser.add_argument('--dataset_address',type=str,default=args.raw_time_series)


#%%
para = parser.parse_args(args=[])
#logging_config_setup(para)

# %%
graph = tf.Graph()
# %%
with graph.as_default():
    initializer = tf.random_uniform_initializer(-para.init_weight,
                                                    para.init_weight)
    data_generator = customDataGenerator(para)
    with tf.variable_scope('model', initializer=initializer):
        model = PolyRNN(para, data_generator)

# %%
with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()
        train(para, sess, model, data_generator)

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:54:48 2021

@author: rejid4996
"""


!nvidia-smi

import tensorflow as tf
tf.__version__

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

session = tf.Session(config=config)
keras.backend.set_session(session)

#%%
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=10,
    allow_soft_placement=True
)

!nvidia-smi
#session = tf.Session(config=config)

with tf.Session(config=config) as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = model.fit(x_train, y_train, epochs=15, batch_size=64)
    model.save_weights('./elmo-model_trial.h5')
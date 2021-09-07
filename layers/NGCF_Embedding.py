import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class NGCF_Embedding(Layer):
    def __init__(self, input_dims, units, mess_dropout, node_dropout_flag, \
        n_fold, n_users, n_items, \
        initer="glorot_uniform", trainable=True):

        super(NGCF_Embedding, self).__init__()
        self.units = units
        self.input_dims = input_dims
        self.initer = initer
        self.trainable = trainable
        self.mess_dropout = mess_dropout
        self.node_dropout_flag = node_dropout_flag
        self.n_fold = n_fold
        self.n_users = n_users
        self.n_items = n_items
        self.weight_shape = (input_dims, units)

        
    def build(self, input_shape):
        print('input shape: ', input_shape)
        self.w_gc = self.add_weight('w_gc', shape=self.weight_shape, initializer=self.initer, trainable=self.trainable)
        self.b_gc = self.add_weight("b_gc", shape=(self.units,), initializer=self.initer, trainable=self.trainable)

        self.w_bi = self.add_weight('w_bi', shape=self.weight_shape, initializer=self.initer, trainable=self.trainable)
        self.b_bi = self.add_weight("b_bi", shape=(self.units,), initializer=self.initer, trainable=self.trainable)
        
        self.w_mlp = self.add_weight('w_mlp', shape=self.weight_shape, initializer=self.initer, trainable=self.trainable)
        self.b_mlp = self.add_weight("b_mlp", shape=(self.units,), initializer=self.initer, trainable=self.trainable)

    def call(self, A_fold_hat, ego_embeddings, all_embeddings, **kwargs):
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

        # sum messages of neighbors.
        side_embeddings = tf.concat(temp_embed, 0)
        # transformed sum messages of neighbors.
        sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.w_gc) + self.b_gc)

        # bi messages of neighbors.
        bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
        # transformed bi messages of neighbors.
        bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.w_bi) + self.b_bi)

        # non-linear activation.
        ego_embeddings = sum_embeddings + bi_embeddings

        # message dropout.
        ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout)

        # normalize the distribution of embeddings.
        norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

        all_embeddings += [norm_embeddings]
        return all_embeddings

    def get_config(self):
        config = self.get_config()
        config.update({"units":self.units})
        config.update({"input_dims":self.input_dims})
        config.update({"initer":self.initer})
        config.update({"trainable":self.trainable})
        config.update({"mess_dropout":self.mess_dropout})
        config.update({"node_dropout_flag":self.node_dropout_flag})
        config.update({"n_fold":self.n_fold})
        config.update({"n_users":self.n_users})
        config.update({"n_items":self.n_items})
        config.update({"weight_shape":self.weight_shape})
        return config
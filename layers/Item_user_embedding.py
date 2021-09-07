import tensorflow as tf
from tensorflow.python.ops.init_ops_v2 import Initializer
from tensorflow.keras.layers import Layer

class Item_user_embedding(Layer):
    def __init__(self, n_users, n_items, emb_dim, initer="glorot_uniform"):
        super().__init__()
        self.user_embedding_weights = self.add_weight(shape=(n_users, emb_dim), initializer=initer, name='user_embedding')
        self.item_embedding_weights = self.add_weight(shape=(n_items, emb_dim), initializer=initer, name='item_embedding')

    def call(self, inputs):
        return tf.concat([self.user_embedding_weights, self.item_embedding_weights], axis=0)

    def get_config(self):
        config = self.get_config()
        # config.update({"units": self.units})
        return config
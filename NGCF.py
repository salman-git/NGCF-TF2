'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from posixpath import basename
import tensorflow as tf
import os
import sys
from tensorflow.keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *
from layers.NGCF_Embedding import NGCF_Embedding
from layers.Item_user_embedding import Item_user_embedding
from tensorflow.keras.optimizers import Adam
import glob

class NGCF(Model):
    def __init__(self, data_config):
        # argument settings
        super(NGCF, self).__init__()
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.node_dropout_flag = args.node_dropout_flag
        self.embed_layers = []
        for i in range(self.n_layers):
            self.embed_layers.append(NGCF_Embedding(self.emb_dim, self.weight_size[i], eval(args.mess_dropout)[i],\
                args.node_dropout_flag, self.n_fold, self.n_users, self.n_items))
        self.user_item_emb_layer = Item_user_embedding(self.n_users, self.n_items, self.emb_dim)
        

    def call(self, users, pos_items, neg_items=None, test=False):

        if self.alg_type in ['ngcf']:
            ua_embeddings, ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            ua_embeddings, ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            ua_embeddings, ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, pos_items)
        if not test:
            neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, neg_items)
            return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        return u_g_embeddings, pos_i_g_embeddings

    def calculate_loss(self, u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings):
        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        mf_loss, emb_loss, reg_loss = self.create_bpr_loss(u_g_embeddings,
                                                                          pos_i_g_embeddings,
                                                                          neg_i_g_embeddings)
        # self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        return mf_loss + emb_loss + reg_loss, mf_loss, emb_loss, reg_loss
        # return 1


    
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = self.user_item_emb_layer(0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            all_embeddings = self.embed_layers[k](A_fold_hat, ego_embeddings, all_embeddings)
            
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.w['user_embedding'], self.w['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.w['W_gc_%d' %k]) + self.w['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.w['user_embedding'], self.w['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.w['W_gc_%d' % k]) + self.w['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.w['W_mlp_%d' %k]) + self.w['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        
        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        maxi = tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        # mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_model(path, config):
    try:
        checkpoints = glob.glob(path + "epoch_*.index")
        epoch_indexes = [ int(re.findall(r'[0-9]+', os.path.basename(p))[0]) for p in checkpoints]
        latest_epoch = max(epoch_indexes)
        model = NGCF(data_config=config)
        #to ensure the model build function is called
        # model([1,2,3], [1,2,3], [1,2,3])

        model.load_weights(path + f"epoch_{latest_epoch}")
        return latest_epoch, model
    except:
        return 0, NGCF(data_config=config)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()
    weights_save_path = None

    if args.save_flag == 1:
        weights_save_path = f'{args.weights_path}weights/{args.dataset}/{model.model_type}/'
        ensureDir(weights_save_path)
    
    start_epoch = 0
    if args.pretrain == 1 and weights_save_path:
        start_epoch, model = load_model(weights_save_path, config)
    else:
        model = NGCF(data_config=config)
    optimizer = Adam(learning_rate=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(start_epoch, args.epoch - args.epoch + 1):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        with tf.GradientTape() as tape:
            for idx in range(n_batch - n_batch + 1):
                users, pos_items, neg_items = data_generator.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items)
                batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = model.calculate_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss
                reg_loss += batch_reg_loss
        
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()
            
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch + 1, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        
        cur_best_pre_0 = 0
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
        # *********************************************************
        if args.save_flag == 1:
            model.save_weights(weights_save_path + f"epoch_{epoch}")

        

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    model.save_weights(save_path)
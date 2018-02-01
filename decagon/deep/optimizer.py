import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class DecagonOptimizer(object):
    def __init__(self, row_embeds, col_embeds, latent_inters, latent_varies,
                 degrees, edge_types, edge_type2dim, placeholders,
                 margin=0.1, neg_sample_weights=1., batch_size=100):
        self.row_embeds = row_embeds
        self.col_embeds = col_embeds
        self.latent_inters = latent_inters
        self.latent_varies = latent_varies
        self.edge_types = edge_types
        self.degrees = degrees
        self.row_edge_type2dim = {i: dim[0][0] for (i, _), dim in edge_type2dim.iteritems()}
        self.edge_type2dim = edge_type2dim
        self.mx_row = np.max([r for rel_list in self.edge_type2dim.values() for r, _ in rel_list])
        self.margin = margin
        self.neg_sample_weights = neg_sample_weights
        self.batch_size = batch_size

        self.inputs = placeholders['batch']
        self.batch_edge_type_idx = placeholders['batch_edge_type_idx']
        self.batch_row_edge_type = placeholders['batch_row_edge_type']
        self.batch_col_edge_type = placeholders['batch_col_edge_type']
        self.row_inputs = tf.squeeze(gather_cols(self.inputs, [0]))
        self.col_inputs = tf.squeeze(gather_cols(self.inputs, [1]))

        labels = tf.reshape(tf.cast(self.row_inputs, dtype=tf.int64), [self.batch_size, 1])
        neg_samples_list = []
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=labels,
                    num_true=1,
                    num_sampled=self.batch_size,
                    unique=False,
                    range_max=len(self.degrees[i][k]),
                    distortion=0.75,
                    unigrams=self.degrees[i][k].tolist())
                neg_samples_list.append(neg_samples)
        self.neg_samples = tf.gather(neg_samples_list, self.batch_edge_type_idx)

        self.preds = self._preds()

        self.outputs = tf.gather_nd(self.preds, self.inputs)
        self.outputs = tf.reshape(self.outputs, [-1])

        a = tf.cast(tf.reshape(self.neg_samples, [self.batch_size, 1]), dtype=tf.int32)
        b = tf.reshape(self.col_inputs, [self.batch_size, 1])
        neg_sample_indices = tf.concat([a, b], axis=1)
        self.neg_outputs = tf.gather_nd(self.preds, neg_sample_indices)
        self.neg_outputs = tf.reshape(self.neg_outputs, [-1])

        self._build()

    def _preds(self):
        row_input = tf.gather(pad(self.row_embeds, self.row_edge_type2dim, self.mx_row), self.batch_row_edge_type)
        col_input = tf.gather(pad(self.col_embeds, self.row_edge_type2dim, self.mx_row), self.batch_col_edge_type)
        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)

        product1 = tf.matmul(row_input, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)
        preds = tf.matmul(product3, tf.transpose(col_input))
        return preds

    def _build(self):
        # self.cost = self._hinge_loss(self.outputs, self.neg_outputs)
        self.cost = self._xent_loss(self.outputs, self.neg_outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

    def _hinge_loss(self, aff, neg_aff):
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 0) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        return loss

    def _xent_loss(self, aff, neg_aff):
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss


def pad(params, row_edge_type2dim, mx_row):
    """Pad a list of 2D tensors to match their shapes.

    Args:
        params: A list of 2D tensor.

    Returns:
        A list of 2D tensors. 
    """
    res = []
    for i, tensor in enumerate(params):
        paddings = tf.constant([[0, mx_row - row_edge_type2dim[i]], [0, 0]])
        padded = tf.pad(tensor, paddings=paddings, mode='CONSTANT')
        res.append(padded)
    return res


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(
            tf.gather(p_flat, i_flat), [p_shape[0], -1])
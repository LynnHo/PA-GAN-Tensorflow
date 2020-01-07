import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_graphics as tfg

import utils


conv = functools.partial(slim.conv2d, activation_fn=None)
dconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
fc = functools.partial(slim.fully_connected, activation_fn=None)


def attention_editor(fa, fb, b, Ge, Gm, m_multi_pre=None):
    rm_none = lambda l: [x for x in l if x is not None]
    n_att = b.shape[-1]

    e_ipt = utils.tile_concat(fb, b)
    e = Ge(e_ipt)

    if m_multi_pre is not None:
        # 1
        # m_multi_pre = tf.image.resize_bicubic(m_multi_pre, [m_multi_pre.shape[1] * 2, m_multi_pre.shape[2] * 2])

        # or 2
        shape = [None, m_multi_pre.shape[1] * 2, m_multi_pre.shape[2] * 2, m_multi_pre.shape[3]]
        m_multi_pre = tfg.image.pyramid.upsample(m_multi_pre, 1)[-1]
        m_multi_pre.set_shape(shape)
    dm_multi_ipt = utils.tile_concat(rm_none([fa, e, m_multi_pre]), b)
    dm_multi = Gm(dm_multi_ipt)

    if m_multi_pre is not None:
        m_multi = m_multi_pre + dm_multi
    else:
        m_multi = dm_multi

    b = tf.reshape(tf.abs(tf.sign(b)), [-1, 1, 1, n_att])
    m = tf.clip_by_value(tf.reduce_sum(b * tf.nn.sigmoid(m_multi), axis=-1, keep_dims=True), 0.0, 1.0)

    fb = m * e + (1 - m) * fa

    return fb, e, m, m_multi


class PAGANG:

    def __call__(self, xa, b, n_downsamplings=5, n_masks=4, dim=64, weight_decay=0.0,
                 norm_name='batch_norm', training=True, scope='PAGANG'):
        MAX_DIM = 1024
        n_att = b.shape[-1]

        conv_ = functools.partial(conv, weights_regularizer=slim.l2_regularizer(weight_decay))
        dconv_ = functools.partial(dconv, weights_regularizer=slim.l2_regularizer(weight_decay))
        norm = utils.get_norm_layer(norm_name, training, updates_collections=None)

        conv_norm_relu = functools.partial(conv_, normalizer_fn=norm, activation_fn=tf.nn.relu)
        dconv_norm_relu = functools.partial(dconv_, normalizer_fn=norm, activation_fn=tf.nn.relu)

        def Gm(x, dim):
            m0 = x
            m0 = conv_norm_relu(m0, dim, 1, 1)

            m1 = x
            m1 = conv_norm_relu(m1, dim, 3, 1)

            m2 = x
            m2 = conv_norm_relu(m2, dim, 3, 1)
            m2 = conv_norm_relu(m2, dim, 3, 1)

            m3 = x
            m3 = conv_norm_relu(m3, dim, 3, 1)
            m3 = conv_norm_relu(m3, dim, 3, 1)
            m3 = conv_norm_relu(m3, dim, 3, 1)

            m = tf.concat([m0, m1, m2, m3], axis=-1)

            m = conv_norm_relu(m, dim * 2, 4, 2)
            m = dconv_(m, n_att, 4, 2)

            return m

        def Ge(x, dim):
            e = x
            e = dconv_norm_relu(e, num_outputs=dim, kernel_size=3, stride=1)
            e = dconv_norm_relu(e, num_outputs=dim, kernel_size=3, stride=2)
            return e

        def Ge_0(x, dim):
            e = x
            e = dconv_norm_relu(e, num_outputs=dim, kernel_size=3, stride=1)
            e = dconv_(e, num_outputs=3, kernel_size=3, stride=2)
            e = tf.nn.tanh(e)
            return e

        # ======================================
        # =              network               =
        # ======================================

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            b = tf.to_float(b)

            # downsamplings
            fa = xa

            fas = [fa]  # fas = [xa, fa_1, fa_2, ..., fa_n_downsamplings]
            for i in range(n_downsamplings):
                d = min(dim * 2**i, MAX_DIM)
                fa = conv_norm_relu(fa, d, 4, 2)
                fas.append(fa)

            # upsamplings
            fb = utils.tile_concat(fa, b)

            # 1 ~ n_downsamplings-n_masks
            for i in range(n_downsamplings - n_masks):
                d = min(int(dim * 2**(n_downsamplings - 2 - i)), MAX_DIM)
                fb = dconv_norm_relu(fb, d, 4, 2)

            # n_downsamplings-n_masks+1 ~ n_downsamplings
            ms = []
            ms_multi = []
            m_multi = None
            for i in range(n_downsamplings - n_masks, n_downsamplings):
                d = min(int(dim * 2**(n_downsamplings - 2 - i)), MAX_DIM)
                if i < n_downsamplings - 1:
                    Ge_ = functools.partial(Ge, dim=d)
                    Gm_ = functools.partial(Gm, dim=d)
                else:
                    Ge_ = functools.partial(Ge_0, dim=d)
                    Gm_ = functools.partial(Gm, dim=d)
                fb, e, mask, m_multi = attention_editor(fas[-2 - i], fb, b, Ge_, Gm_, m_multi_pre=m_multi)
                ms.append(mask)
                ms_multi.append(m_multi)

            x = fb

        # variables and update operations
        self.variables = tf.global_variables(scope)
        self.trainable_variables = tf.trainable_variables(scope)
        self.reg_losses = tf.losses.get_regularization_losses(scope)

        return x, e, ms, ms_multi


class PAGAND:

    def __call__(self, x, n_atts, n_downsamplings=5, dim=64, fc_dim=1024, weight_decay=0.0,
                 norm_name='layer_norm', training=True, scope='PAGAND'):
        MAX_DIM = 1024

        conv_ = functools.partial(conv, weights_regularizer=slim.l2_regularizer(weight_decay))
        fc_ = functools.partial(fc, weights_regularizer=slim.l2_regularizer(weight_decay))
        norm = utils.get_norm_layer(norm_name, training, updates_collections=None)

        conv_norm_lrelu = functools.partial(conv_, normalizer_fn=norm, activation_fn=tf.nn.leaky_relu)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            z = x
            for i in range(n_downsamplings):
                d = min(dim * 2**i, MAX_DIM)
                z = conv_norm_lrelu(z, d, 4, 2)
            z = slim.flatten(z)

            logit_gan = tf.nn.leaky_relu(fc_(z, fc_dim))
            logit_gan = fc_(logit_gan, 1)

            logit_att = tf.nn.leaky_relu(fc_(z, fc_dim))
            logit_att = fc_(logit_att, n_atts)

        # variables and update operations
        self.variables = tf.global_variables(scope)
        self.trainable_variables = tf.trainable_variables(scope)
        self.reg_losses = tf.losses.get_regularization_losses(scope)

        return logit_gan, logit_att


def overlap_loss_fn(ms_multi, att_names):
    # ======================================
    # =        customized relation         =
    # ======================================

    full_overlap_pairs = [
        # ('Black_Hair', 'Blond_Hair'),
        # ('Black_Hair', 'Brown_Hair'),

        # ('Blond_Hair', 'Brown_Hair')
    ]

    non_overlap_pairs = [
        # ('Bald', 'Bushy_Eyebrows'),
        # ('Bald', 'Eyeglasses'),
        ('Bald', 'Mouth_Slightly_Open'),
        ('Bald', 'Mustache'),
        ('Bald', 'No_Beard'),

        ('Bangs', 'Mouth_Slightly_Open'),
        ('Bangs', 'Mustache'),
        ('Bangs', 'No_Beard'),

        ('Black_Hair', 'Mouth_Slightly_Open'),
        ('Black_Hair', 'Mustache'),
        ('Black_Hair', 'No_Beard'),

        ('Blond_Hair', 'Mouth_Slightly_Open'),
        ('Blond_Hair', 'Mustache'),
        ('Blond_Hair', 'No_Beard'),

        ('Brown_Hair', 'Mouth_Slightly_Open'),
        ('Brown_Hair', 'Mustache'),
        ('Brown_Hair', 'No_Beard'),

        # ('Bushy_Eyebrows', 'Mouth_Slightly_Open'),
        ('Bushy_Eyebrows', 'Mustache'),
        ('Bushy_Eyebrows', 'No_Beard'),

        # ('Eyeglasses', 'Mouth_Slightly_Open'),
        ('Eyeglasses', 'Mustache'),
        ('Eyeglasses', 'No_Beard'),
    ]

    # ======================================
    # =                 losses             =
    # ======================================

    full_overlap_pair_loss = tf.constant(0.0)
    for p in full_overlap_pairs:
        id1 = att_names.index(p[0])
        id2 = att_names.index(p[1])
        for m in ms_multi[-1:]:
            m1 = m[..., id1]
            m2 = m[..., id2]
            full_overlap_pair_loss += tf.losses.absolute_difference(m1, m2)

    non_overlap_pair_loss = tf.constant(0.0)
    for p in non_overlap_pairs:
        id1 = att_names.index(p[0])
        id2 = att_names.index(p[1])
        for m in ms_multi[-1:]:
            m1 = m[..., id1]
            m2 = m[..., id2]
            non_overlap_pair_loss += tf.reduce_mean(tf.nn.sigmoid(m1) * tf.nn.sigmoid(m2))

    return full_overlap_pair_loss, non_overlap_pair_loss

import functools
import traceback

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tfprob
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

default_att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
py.arg('--att_names', choices=data.ATT_ID.keys(), nargs='+', default=default_att_names)

py.arg('--img_dir', default='./data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data')
py.arg('--train_label_path', default='./data/img_celeba/train_label.txt')
py.arg('--val_label_path', default='./data/img_celeba/val_label.txt')
py.arg('--load_size', type=int, default=143)
py.arg('--crop_size', type=int, default=128)

py.arg('--n_epochs', type=int, default=60)
py.arg('--epoch_start_decay', type=int, default=30)
py.arg('--batch_size', type=int, default=32)
py.arg('--learning_rate', type=float, default=2e-4)
py.arg('--beta_1', type=float, default=0.5)

py.arg('--dim', type=int, default=64)

py.arg('--n_d', type=int, default=5)  # # d updates per g update
py.arg('--adversarial_loss_mode', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'], default='wgan')
py.arg('--gradient_penalty_mode', choices=['none', '1-gp', '0-gp', 'lp'], default='lp')
py.arg('--gradient_penalty_sample_mode', choices=['line', 'real', 'fake', 'dragan'], default='line')
py.arg('--d_gradient_penalty_weight', type=float, default=10.0)
py.arg('--d_attribute_loss_weight', type=float, default=1.0)
py.arg('--g_attribute_loss_weight', type=float, default=20.0)
py.arg('--g_spasity_loss_weight', type=float, default=0.05)
py.arg('--g_full_overlap_mask_pair_loss_weight', type=float, default=1.0)
py.arg('--g_non_overlap_mask_pair_loss_weight', type=float, default=1.0)
py.arg('--weight_decay', type=float, default=0.0)

py.arg('--n_samples', type=int, default=16)
py.arg('--test_int', type=float, default=1.5)

py.arg('--experiment_name', default='default')
args = py.args()

# output_dir
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# others
n_atts = len(args.att_names)

sess = tl.session()
sess.__enter__()  # make default


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# data
train_dataset, len_train_dataset = data.make_celeba_dataset(args.img_dir, args.train_label_path, args.att_names, args.batch_size,
                                                            load_size=args.load_size, crop_size=args.crop_size,
                                                            training=True, shuffle=True, repeat=None)
val_dataset, len_val_dataset = data.make_celeba_dataset(args.img_dir, args.val_label_path, args.att_names, args.n_samples,
                                                        load_size=args.load_size, crop_size=args.crop_size,
                                                        training=False, shuffle=False, repeat=None)
train_iter = train_dataset.make_one_shot_iterator()
val_iter = val_dataset.make_one_shot_iterator()

# model
G = functools.partial(module.PAGANG(), dim=args.dim, weight_decay=args.weight_decay)
D = functools.partial(module.PAGAND(), n_atts=n_atts, dim=args.dim, weight_decay=args.weight_decay)

# loss functions
d_loss_fn, g_loss_fn = tfprob.get_adversarial_losses_fn(args.adversarial_loss_mode)


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def D_train_graph():
    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    lr = tf.placeholder(dtype=tf.float32, shape=[])

    xa, a = train_iter.get_next()
    b = tf.random_shuffle(a)
    a_ = a * 2 - 1
    b_ = b * 2 - 1

    # generate
    xb, _, ms, _ = G(xa, b_ - a_)

    # discriminate
    xa_logit_gan, xa_logit_att = D(xa)
    xb_logit_gan, xb_logit_att = D(xb)

    # discriminator losses
    xa_loss_gan, xb_loss_gan = d_loss_fn(xa_logit_gan, xb_logit_gan)
    gp = tfprob.gradient_penalty(lambda x: D(x)[0], xa, xb, args.gradient_penalty_mode, args.gradient_penalty_sample_mode)
    xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)
    reg_loss = tf.reduce_sum(D.func.reg_losses)

    loss = (xa_loss_gan + xb_loss_gan +
            gp * args.d_gradient_penalty_weight +
            xa_loss_att * args.d_attribute_loss_weight +
            reg_loss)

    # optim
    step_cnt, _ = tl.counter()
    step = tf.train.AdamOptimizer(lr, beta1=args.beta_1).minimize(loss, global_step=step_cnt, var_list=D.func.trainable_variables)

    # summary
    with tf.contrib.summary.create_file_writer('./output/%s/summaries/D' % args.experiment_name).as_default(),\
            tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_cnt):
        summary = [
            tl.summary_v2({
                'loss_gan': xa_loss_gan + xb_loss_gan,
                'gp': gp,
                'xa_loss_att': xa_loss_att,
                'reg_loss': reg_loss
            }, step=step_cnt, name='D'),
            tl.summary_v2({'lr': lr}, step=step_cnt, name='learning_rate')
        ]

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        sess.run([step, summary], feed_dict={lr: pl_ipts['lr']})

    return run


def G_train_graph():
    # ======================================
    # =                 graph              =
    # ======================================

    # placeholders & inputs
    lr = tf.placeholder(dtype=tf.float32, shape=[])

    xa, a = train_iter.get_next()
    b = tf.random_shuffle(a)
    a_ = a * 2 - 1
    b_ = b * 2 - 1

    # generate
    xb, _, ms, ms_multi = G(xa, b_ - a_)

    # discriminate
    xb_logit_gan, xb_logit_att = D(xb)

    # generator losses
    xb_loss_gan = g_loss_fn(xb_logit_gan)
    xb_loss_att = tf.losses.sigmoid_cross_entropy(b, xb_logit_att)
    spasity_loss = tf.reduce_sum([tf.reduce_mean(m) * w for m, w in zip(ms, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])])
    full_overlap_mask_pair_loss, non_overlap_mask_pair_loss = module.overlap_loss_fn(ms_multi, args.att_names)
    reg_loss = tf.reduce_sum(G.func.reg_losses)

    loss = (xb_loss_gan +
            xb_loss_att * args.g_attribute_loss_weight +
            spasity_loss * args.g_spasity_loss_weight +
            full_overlap_mask_pair_loss * args.g_full_overlap_mask_pair_loss_weight +
            non_overlap_mask_pair_loss * args.g_non_overlap_mask_pair_loss_weight +
            reg_loss)

    # optim
    step_cnt, _ = tl.counter()
    step = tf.train.AdamOptimizer(lr, beta1=args.beta_1).minimize(loss, global_step=step_cnt, var_list=G.func.trainable_variables)

    # summary
    with tf.contrib.summary.create_file_writer('./output/%s/summaries/G' % args.experiment_name).as_default(),\
            tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_cnt):
        summary = tl.summary_v2({
            'xb_loss_gan': xb_loss_gan,
            'xb_loss_att': xb_loss_att,
            'spasity_loss': spasity_loss,
            'full_overlap_mask_pair_loss': full_overlap_mask_pair_loss,
            'non_overlap_mask_pair_loss': non_overlap_mask_pair_loss,
            'reg_loss': reg_loss
        }, step=step_cnt, name='G')

    # ======================================
    # =           generator size           =
    # ======================================

    n_params, n_bytes = tl.count_parameters(G.func.variables)
    print('Generator Size: n_parameters = %d = %.2fMB' % (n_params, n_bytes / 1024 / 1024))

    # ======================================
    # =            run function            =
    # ======================================

    def run(**pl_ipts):
        sess.run([step, summary], feed_dict={lr: pl_ipts['lr']})

    return run


def sample_graph():
    # ======================================
    # =               graph                =
    # ======================================

    # placeholders & inputs
    xa = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
    a_ = tf.placeholder(tf.float32, shape=[None, n_atts])
    b_ = tf.placeholder(tf.float32, shape=[None, n_atts])

    # sample graph
    x, e, ms, _ = G(xa, b_ - a_, training=False)

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_training' % args.experiment_name
    py.mkdir(save_dir)

    def run(epoch, iter):
        # data for sampling
        xa_ipt, a_ipt = sess.run(val_iter.get_next())
        b_ipt_list = [a_ipt]  # the first is for reconstruction
        for i in range(n_atts):
            tmp = np.array(a_ipt, copy=True)
            tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
            tmp = data.check_attribute_conflict(tmp, args.att_names[i], args.att_names)
            b_ipt_list.append(tmp)

        x_opt_list = [xa_ipt]
        e_opt_list = [np.full_like(xa_ipt, -1.0)]
        ms_opt_list = []
        a__ipt = a_ipt * 2 - 1
        for i, b_ipt in enumerate(b_ipt_list):
            b__ipt = (b_ipt * 2 - 1).astype(np.float32)  # !!!
            if i > 0:   # i == 0 is for reconstruction
                b__ipt[..., i - 1] = b__ipt[..., i - 1] * args.test_int
            x_opt, e_opt, ms_opt = sess.run([x, e, ms], feed_dict={xa: xa_ipt, a_: a__ipt, b_: b__ipt})
            x_opt_list.append(x_opt)
            e_opt_list.append(e_opt)
            ms_opt_list.append(ms_opt)

        # save sample
        sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
        sample = np.reshape(sample, (-1, sample.shape[2] * sample.shape[3], sample.shape[4]))

        # resize all masks to the same size
        for ms_opt in ms_opt_list:  # attribute axis
            for i, m_opt in enumerate(ms_opt):  # mask level axis
                m_opt_resized = []
                for m_j_opt in m_opt:  # batch axis
                    m_opt_resized.append(im.imresize(m_j_opt * 2 - 1, (args.crop_size, args.crop_size)))
                ms_opt[i] = np.concatenate([np.array(m_opt_resized)] * 3, axis=-1)
        ms_opt_list = [np.full_like(ms_opt_list[0], -1.0)] + ms_opt_list
        ms_opt_list = list(np.transpose(ms_opt_list, (1, 0, 2, 3, 4, 5)))[::-1]
        sample_m = np.transpose([x_opt_list, e_opt_list] + ms_opt_list, (2, 0, 3, 1, 4, 5))
        sample_m = np.reshape(sample_m, (-1, sample_m.shape[3] * sample_m.shape[4], sample_m.shape[5]))
        im.imwrite(np.concatenate((sample, sample_m)), '%s/Epoch-%d_Iter-%d.jpg' % (save_dir, epoch, iter))

    return run


D_train_step = D_train_graph()
G_train_step = G_train_graph()
sample = sample_graph()


# ==============================================================================
# =                                   train                                    =
# ==============================================================================

# step counter
step_cnt, update_cnt = tl.counter()

# checkpoint
checkpoint = tl.Checkpoint(
    {v.name: v for v in tf.global_variables()},
    py.join(output_dir, 'checkpoints'),
    max_to_keep=1
)
checkpoint.restore().initialize_or_restore()

# summary
sess.run(tf.contrib.summary.summary_writer_initializer_op())

# learning rate schedule
lr_fn = tl.LinearDecayLR(args.learning_rate, args.n_epochs, args.epoch_start_decay)

# train
try:
    for ep in tqdm.trange(args.n_epochs, desc='Epoch Loop'):
        # learning rate
        lr_ipt = lr_fn(ep)

        for it in tqdm.trange(len_train_dataset, desc='Inner Epoch Loop'):
            if it + ep * len_train_dataset < sess.run(step_cnt):
                continue
            step = sess.run(update_cnt)

            # train D
            if step % (args.n_d + 1) != 0:
                D_train_step(lr=lr_ipt)
            # train G
            else:
                G_train_step(lr=lr_ipt)

            # save
            if step % (1000 * (args.n_d + 1)) == 0:
                checkpoint.save(step)

            # sample
            if step % (100 * (args.n_d + 1)) == 0:
                sample(ep, it)
except Exception:
    traceback.print_exc()
finally:
    checkpoint.save(step)
    sess.close()

import functools
import os

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--img_dir', default='./data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data')
py.arg('--test_label_path', default='./data/img_celeba/test_label.txt')
py.arg('--with_mask', default=False)
py.arg('--test_att_names', choices=data.ATT_ID.keys(), nargs='+', default=['Bangs', 'Mustache'])
py.arg('--test_ints', type=float, nargs='+', default=2)

py.arg('--experiment_name', default='default')
args_ = py.args()

# output_dir
output_dir = py.join('output', args_.experiment_name)

# save settings
args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
args.__dict__.update(args_.__dict__)

# others
n_atts = len(args.att_names)
if not isinstance(args.test_ints, list):
    args.test_ints = [args.test_ints] * len(args.test_att_names)
elif len(args.test_ints) == 1:
    args.test_ints = args.test_ints * len(args.test_att_names)

sess = tl.session()
sess.__enter__()  # make default


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# data
test_dataset, len_test_dataset = data.make_celeba_dataset(args.img_dir, args.test_label_path, args.att_names, args.n_samples,
                                                          load_size=args.load_size, crop_size=args.crop_size,
                                                          training=False, drop_remainder=False, shuffle=False, repeat=None)
test_iter = test_dataset.make_one_shot_iterator()


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def sample_graph():
    # ======================================
    # =               graph                =
    # ======================================

    if not os.path.exists(py.join(output_dir, 'generator.pb')):
        # model
        G = functools.partial(module.PAGANG(), dim=args.dim, weight_decay=args.weight_decay)

        # placeholders & inputs
        xa = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
        a_ = tf.placeholder(tf.float32, shape=[None, n_atts])
        b_ = tf.placeholder(tf.float32, shape=[None, n_atts])

        # sample graph
        x, e, ms, _ = G(xa, b_ - a_, training=False)
    else:
        # load freezed model
        with tf.gfile.GFile(py.join(output_dir, 'generator.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='generator')

        # placeholders & inputs
        xa = sess.graph.get_tensor_by_name('generator/xa:0')
        a_ = sess.graph.get_tensor_by_name('generator/a_:0')
        b_ = sess.graph.get_tensor_by_name('generator/b_:0')

        # sample graph
        x = sess.graph.get_tensor_by_name('generator/xb:0')
        e = sess.graph.get_tensor_by_name('generator/e:0')
        ms = sess.graph.get_operation_by_name('generator/ms').outputs

    # ======================================
    # =            run function            =
    # ======================================

    if args.with_mask:
        save_dir = './output/%s/samples_testing_multi_with_mask' % args.experiment_name
    else:
        save_dir = './output/%s/samples_testing_multi' % args.experiment_name
    tmp = ''
    for test_att_name, test_int in zip(args.test_att_names, args.test_ints):
        tmp += '_%s_%s' % (test_att_name, '{:g}'.format(test_int))
    save_dir = py.join(save_dir, tmp[1:])
    py.mkdir(save_dir)

    def run():
        cnt = 0
        for _ in tqdm.trange(len_test_dataset):
            # data for sampling
            xa_ipt, a_ipt = sess.run(test_iter.get_next())
            b_ipt = np.copy(a_ipt)
            for test_att_name in args.test_att_names:
                i = args.att_names.index(test_att_name)
                b_ipt[..., i] = 1 - b_ipt[..., i]
                b_ipt = data.check_attribute_conflict(b_ipt, test_att_name, args.att_names)

            a__ipt = a_ipt * 2 - 1
            b__ipt = (b_ipt * 2 - 1).astype(np.float32)  # !!!
            for test_att_name, test_int in zip(args.test_att_names, args.test_ints):
                i = args.att_names.index(test_att_name)
                b__ipt[..., i] = b__ipt[..., i] * test_int

            x_opt_list = [xa_ipt]
            e_opt_list = [np.full_like(xa_ipt, -1.0)]
            ms_opt_list = []
            x_opt, e_opt, ms_opt = sess.run([x, e, ms], feed_dict={xa: xa_ipt, a_: a__ipt, b_: b__ipt})
            x_opt_list.append(x_opt)
            e_opt_list.append(e_opt)
            ms_opt_list.append(ms_opt)

            if args.with_mask:
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
            else:
                sample_m = np.transpose([x_opt_list], (2, 0, 3, 1, 4, 5))
            sample_m = np.reshape(sample_m, (sample_m.shape[0], -1, sample_m.shape[3] * sample_m.shape[4], sample_m.shape[5]))

            for s in sample_m:
                cnt += 1
                im.imwrite(s, '%s/%d.jpg' % (save_dir, cnt))

    return run


sample = sample_graph()


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# checkpoint
if not os.path.exists(py.join(output_dir, 'generator.pb')):
    checkpoint = tl.Checkpoint(
        {v.name: v for v in tf.global_variables()},
        py.join(output_dir, 'checkpoints'),
        max_to_keep=1
    )
    checkpoint.restore().run_restore_ops()

sample()

sess.close()

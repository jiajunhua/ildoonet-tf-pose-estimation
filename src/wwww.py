import numpy as np
import tensorflow as tf

from src.common import read_imgfile
from src.network_mobilenet_try2 import MobilenetNetworkTry2
from src.slim.nets import mobilenet_v1_base, _CONV_DEFS

if __name__ == '__main__':
    mode = 'mine'

    img = read_imgfile('./images/cat.jpg', 224, 224)

    input_node = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='image')
    target = 'Conv2d_11'

    vars = None
    if mode == 'mine':
        net = MobilenetNetworkTry2({'image': input_node}, trainable=True)
        t = net.get_tensor(target)
        vars = net.restorable_variables()
    elif mode == 'google':
        input_node = tf.divide(input_node, 255.0, name='i_divide')
        input_node = tf.subtract(input_node, 0.5, name='i_subtract')
        input_node = tf.multiply(input_node, 2.0, name='i_multiply')
        t, _ = mobilenet_v1_base(input_node, final_endpoint=target + '_pointwise', conv_defs=_CONV_DEFS)
    else:
        raise Exception()

    for op in tf.get_default_graph().get_operations():
        print(op.name)

    print('----')

    w_c0 = None
    for tv in tf.trainable_variables():
        # print(tv)
        if 'Conv2d_0/weights:0' in tv.name:
            w_c0 = tv

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loader = tf.train.Saver(vars)
        loader.restore(sess, './models/pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt')

        output = sess.run(t, feed_dict={input_node: [img]})
        weights = sess.run(w_c0)
        print(np.max(output))
        print(np.sum(output))
        pass

import network_base
import tensorflow as tf


class MobilenetNetwork(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0):
        self.conv_width = conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)

        with tf.variable_scope(None, 'MobilenetV1'):
            (self.feed('image')
             .conv(3, 3, depth(32), 2, 2, biased=False, name='Conv2d_0', trainable=self.trainable)
             .separable_conv(3, 3, depth(64), 1, name='Conv2d_1', trainable=self.trainable)
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_2', trainable=self.trainable)
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_3', trainable=self.trainable)
             .separable_conv(3, 3, depth(256), 2, name='Conv2d_4', trainable=self.trainable)
             .separable_conv(3, 3, depth(256), 1, name='Conv2d_5', trainable=self.trainable)
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_6', trainable=self.trainable)
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_7', trainable=self.trainable)
             .separable_conv(3, 3, depth(512), 2, name='Conv2d_8', trainable=self.trainable)
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_9', trainable=self.trainable)
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_10', trainable=self.trainable)
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_11', trainable=self.trainable)
             # .separable_conv(3, 3, depth(1024), 2, name='Conv2d_12', trainable=self.trainable)
             # .separable_conv(3, 3, depth(1024), 1, name='Conv2d_13', trainable=self.trainable)
             )

        feature_lv = 'Conv2d_11'
        with tf.variable_scope(None, 'Openpose'):
            prefix = 'MConv_Stage1'
            (self.feed(feature_lv)
             .separable_conv(3, 3, depth(128), 1, name=prefix + '_L1_1', trainable=self.trainable)
             .separable_conv(3, 3, depth(128), 1, name=prefix + '_L1_2', trainable=self.trainable)
             .separable_conv(3, 3, depth(128), 1, name=prefix + '_L1_3', trainable=self.trainable)
             .separable_conv(1, 1, depth(128), 1, name=prefix + '_L1_4', trainable=self.trainable)
             # .conv(1, 1, depth(160), 1, 1, biased=False, name=prefix + '_L1_4', trainable=self.trainable)
             .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5', trainable=self.trainable))
             # .conv(1, 1, 38, 1, 1, biased=False, name=prefix + '_L1_5', trainable=self.trainable))

            (self.feed(feature_lv)
             .separable_conv(3, 3, depth(128), 1, name=prefix + '_L2_1', trainable=self.trainable)
             .separable_conv(3, 3, depth(128), 1, name=prefix + '_L2_2', trainable=self.trainable)
             .separable_conv(3, 3, depth(128), 1, name=prefix + '_L2_3', trainable=self.trainable)
             .separable_conv(1, 1, depth(128), 1, name=prefix + '_L2_4', trainable=self.trainable)
             # .conv(1, 1, depth(160), 1, 1, biased=False, name=prefix + '_L2_4', trainable=self.trainable)
             .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5', trainable=self.trainable))
             # .conv(1, 1, 19, 1, 1, biased=False, name=prefix + '_L2_5', trainable=self.trainable))

            for stage_id in range(5):
                prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
                prefix = 'MConv_Stage%d' % (stage_id + 2)
                (self.feed(prefix_prev + '_L1_5',
                           prefix_prev + '_L2_5',
                           feature_lv)
                 .concat(3, name=prefix + '_concat')
                 .separable_conv(3, 3, depth(128), 1, name=prefix + '_L1_1', trainable=self.trainable)
                 .separable_conv(3, 3, depth(128), 1, name=prefix + '_L1_2', trainable=self.trainable)
                 .separable_conv(3, 3, depth(128), 1, name=prefix + '_L1_3', trainable=self.trainable)
                 .separable_conv(1, 1, depth(128), 1, name=prefix + '_L1_4', trainable=self.trainable)
                 # .conv(1, 1, depth(160), 1, 1, biased=False, name=prefix + '_L1_4', trainable=self.trainable)
                 .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5', trainable=self.trainable))
                 # .conv(1, 1, 38, 1, 1, biased=False, name=prefix + '_L1_5', trainable=self.trainable))

                (self.feed(prefix + '_concat')
                 .separable_conv(3, 3, depth(128), 1, name=prefix + '_L2_1', trainable=self.trainable)
                 .separable_conv(3, 3, depth(128), 1, name=prefix + '_L2_2', trainable=self.trainable)
                 .separable_conv(3, 3, depth(128), 1, name=prefix + '_L2_3', trainable=self.trainable)
                 .separable_conv(1, 1, depth(128), 1, name=prefix + '_L2_4', trainable=self.trainable)
                 # .conv(1, 1, depth(160), 1, 1, biased=False, name=prefix + '_L2_4', trainable=self.trainable)
                 .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5', trainable=self.trainable))
                 # .conv(1, 1, 19, 1, 1, biased=False, name=prefix + '_L2_5', trainable=self.trainable))

            # final result
            (self.feed('MConv_Stage6_L2_5',
                       'MConv_Stage6_L1_5')
             .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
        l1s = []
        l2s = []
        for layer_name in sorted(self.layers.keys()):
            if '_L1_5' in layer_name:
                l1s.append(self.layers[layer_name])
            if '_L2_5' in layer_name:
                l2s.append(self.layers[layer_name])

        return l1s, l2s

    def loss_last(self):
        return self.get_output('MConv_Stage6_L1_5'), self.get_output('MConv_Stage6_L2_5')

    def restorable_variables(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'MobilenetV1/Conv2d' in v.op.name and
              'RMSProp' not in v.op.name and
              'bias' not in v.op.name
              }
        return vs

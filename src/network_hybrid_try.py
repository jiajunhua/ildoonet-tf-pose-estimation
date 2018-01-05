import tensorflow as tf

from src import network_base


class HybridNetworkTry(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width2=1.0):
        self.conv_width2 = conv_width2
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        (self.feed('image')
         .normalize_vgg(name='preprocess')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, name='pool1_stage1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, name='pool2_stage1')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .conv(3, 3, 256, 1, 1, name='conv3_4')
         .max_pool(2, 2, 2, 2, name='pool3_stage1')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
         .conv(3, 3, 128, 1, 1, name='conv4_4_CPM'))

        feature_lv = 'conv4_4_CPM'
        prefix = 'MConv_Stage1'
        (self.feed(feature_lv)
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
         .convb(1, 1, depth2(512), 1, name=prefix + '_L1_4')
         .convb(1, 1, 38, 1, relu=False, set_tanh=True, name=prefix + '_L1_5'))

        (self.feed(feature_lv)
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
         .convb(1, 1, depth2(512), 1, name=prefix + '_L2_4')
         .convb(1, 1, 19, 1, relu=False, set_tanh=True, name=prefix + '_L2_5'))

        for stage_id in range(5):
            prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
            prefix = 'MConv_Stage%d' % (stage_id + 2)
            (self.feed(prefix_prev + '_L1_5',
                       prefix_prev + '_L2_5',
                       feature_lv)
             .concat(3, name=prefix + '_concat')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_1')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_2')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_3')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_3_2')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L1_3_3')
             .convb(1, 1, depth2(128), 1, name=prefix + '_L1_4')
             .convb(1, 1, 38, 1, relu=False, set_tanh=True, name=prefix + '_L1_5'))

            (self.feed(prefix + '_concat')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_1')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_2')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_3')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_3_2')
             .separable_conv(7, 7, depth2(128), 1, name=prefix + '_L2_3_3')
             .convb(1, 1, depth2(128), 1, name=prefix + '_L2_4')
             .convb(1, 1, 19, 1, relu=False, set_tanh=True, name=prefix + '_L2_5'))

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
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and 'Ada' not in v.op.name
              }
        return vs

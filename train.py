import argparse
import logging
import os
import time

import datetime
import tensorflow as tf

from network_mobilenet import MobilenetNetwork
from pose_dataset import get_dataflow_batch, DataFlowToQueue


logging.basicConfig(level=logging.DEBUG, format='[lmdb_dataset] %(asctime)s %(levelname)s %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet_1.0', help='model name')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco-pose-estimation-lmdb/')
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.00004)
    parser.add_argument('--modelpath', type=str, default='/date/private/tf-openpose-mobilenet_1.0/')

    args = parser.parse_args()

    # define input placeholder
    input_wh = 368
    input_node = tf.placeholder(tf.float32, shape=(args.batchsize, input_wh, input_wh, 3), name='image')

    # define output placeholder
    if args.model in ['mobilenet_1.0', 'mobilenet_0.75', 'mobilenet_0.50']:
        output_size = 46
    else:
        raise Exception('Invalid Mode.')

    vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_size, output_size, 38), name='vectmap')
    heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_size, output_size, 19), name='heatmap')

    # prepare data
    df = get_dataflow_batch(args.datapath, True, args.batchsize)
    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize)
    enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
    q_inp, q_heat, q_vect = enqueuer.dequeue()

    # define model
    if args.model == 'mobilenet_1.0':
        net = MobilenetNetwork({'image': q_inp}, conv_width=1.0)
        pretrain_path = './models/pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
    elif args.model == 'mobilenet_0.75':
        net = MobilenetNetwork({'image': q_inp}, conv_width=0.75)
        pretrain_path = './models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
    elif args.model == 'mobilenet_0.50':
        net = MobilenetNetwork({'image': q_inp}, conv_width=0.50)
        pretrain_path = './models/pretrained/mobilenet_v1_0.50_224_2017_06_14/mobilenet_v1_0.50_224.ckpt'
    else:
        raise Exception('Invalid Mode.')
    output_vectmap = net.get_output('MConv_Stage6_L1_5')
    output_heatmap = net.get_output('MConv_Stage6_L2_5')

    # define loss
    l1s, l2s = net.loss_l1_l2()
    losses = []
    logging.info('loss # = %d %d' % (len(l1s), len(l2s)))
    for l1 in l1s:
        loss = tf.nn.l2_loss(l1 - q_vect, name='loss_' + l1.name.replace(':0', ''))
        losses.append(loss)
    for l2 in l2s:
        loss = tf.nn.l2_loss(l2 - q_heat, name='loss_' + l2.name.replace(':0', ''))
        losses.append(loss)

    total_loss = tf.reduce_mean(losses)
    total_ll_loss = tf.reduce_mean([
        tf.nn.l2_loss(net.loss_last()[0] - q_vect),
        tf.nn.l2_loss(net.loss_last()[1] - q_heat)
    ])

    # define optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = args.lr
    momentum = 0.9
    max_epoch = 50
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decay_steps=10000, decay_rate=0.90, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    train_op = optimizer.minimize(total_loss, global_step)

    # define summary
    sample_train = tf.placeholder(tf.float32, shape=(1, 640, 640, 3))
    sample_valid_gt = tf.placeholder(tf.float32, shape=(1, 640, 640, 3))
    sample_valid_predict = tf.placeholder(tf.float32, shape=(1, 640, 640, 3))
    # tf.summary.image('training sample', sample_train, 1)
    # tf.summary.image('validation ground truth', sample_valid_gt, 1)
    # tf.summary.image('validation prediction', sample_valid_predict, 1)
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_ll_loss)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if pretrain_path:
            logging.info('Restore pretrained weights...')
            logging.info(net.restorable_variables().keys())
            loader = tf.train.Saver(net.restorable_variables())
            loader.restore(sess, pretrain_path)
            logging.info('Restore pretrained weights...Done')

        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        training_name = '{}_{}_batch:{}_lr:{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            args.model,
            args.batchsize,
            args.lr
        )
        file_writer = tf.summary.FileWriter('/data/private/tensorboard-openpose/{}/'.format(training_name), sess.graph)

        logging.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        step_per_epoch = 120000 / args.batchsize
        while True:
            _, gs_num = sess.run([train_op, global_step])

            if gs_num > step_per_epoch * max_epoch:
                break

            if gs_num - last_gs_num >= 100:
                train_loss, train_loss_ll, lr_val, summary = sess.run([total_loss, total_ll_loss, learning_rate, merged_summary_op])

                # log of training loss / accuracy
                batch_per_sec = gs_num / (time.time() - time_started)
                logging.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll))
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num - last_gs_num2 >= 1000:
                average_loss = average_loss_ll = 0
                total_cnt = 0
                df_valid.reset_state()
                gen_val = df_valid.get_data()
                while True:
                    # log of test accuracy
                    try:
                        images_test, heatmaps, vectmaps = next(gen_val)
                    except StopIteration:
                        break

                    lss, lss_ll, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, total_ll_loss, output_vectmap, output_heatmap],
                        feed_dict={input_node: images_test, vectmap_node: vectmaps, heatmap_node: heatmaps}
                    )
                    average_loss += lss * len(images_test)
                    average_loss_ll += lss_ll * len(images_test)
                    total_cnt += len(images_test)

                logging.info('validation(%d) loss=%f, loss_ll=%f' % (total_cnt, average_loss / total_cnt, average_loss_ll / total_cnt))
                last_gs_num2 = gs_num

            if gs_num > 0 and gs_num % 10000 == 0:
                saver.save(sess, os.path.join(args.modelpath, 'model'), global_step=global_step)

        saver.save(sess, os.path.join(args.modelpath, 'model_final'), global_step=global_step)
    logging.info('optimization finished. %f' % (time.time() - time_started))

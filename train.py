import argparse
import logging
import os
import time
import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from network_cmu import CmuNetwork
from network_mobilenet import MobilenetNetwork
from pose_augment import set_network_input_wh
from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPoseLMDB
from tensorpack.dataflow.remote import send_dataflow_zmq, RemoteDataZMQ

logging.basicConfig(level=logging.DEBUG, format='[lmdb_dataset] %(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet_1.0', help='model name')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco-pose-estimation-lmdb/')
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-mobilenet_1.0/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    args = parser.parse_args()

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    output_w = args.input_width // 8
    output_h = args.input_height // 8

    input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
    vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 38), name='vectmap')
    heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 19), name='heatmap')

    # prepare data
    if not args.remote_data:
        df = get_dataflow_batch(args.datapath, True, args.batchsize)
    else:
        df = RemoteDataZMQ(args.remote_data, hwm=5)
    enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
    q_inp, q_heat, q_vect = enqueuer.dequeue()

    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize)
    df_valid.reset_state()
    validation_cache = []
    for images_test, heatmaps, vectmaps in df_valid.get_data():
        validation_cache.append((images_test, heatmaps, vectmaps))

    val_image = cv2.imread('./images/person1.jpg')
    val_image = cv2.resize(val_image, (args.input_width, args.input_height))
    val_image = val_image.astype(float)
    val_image = val_image * (2.0 / 255.0) - 1.0

    # define model for multi-gpu
    q_inp_split = tf.split(q_inp, args.gpus)
    output_vectmap = []
    output_heatmap = []
    vectmap_losses = []
    heatmap_losses = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                if args.model == 'mobilenet_1.0':
                    net = MobilenetNetwork({'image': q_inp_split[gpu_id]}, conv_width=1.0)
                    pretrain_path = './models/pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
                    last_layer = 'MConv_Stage6_L{aux}_5'
                elif args.model == 'mobilenet_0.75':
                    net = MobilenetNetwork({'image': q_inp_split[gpu_id]}, conv_width=0.75)
                    pretrain_path = './models/pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
                    last_layer = 'MConv_Stage6_L{aux}_5'
                elif args.model == 'mobilenet_0.50':
                    net = MobilenetNetwork({'image': q_inp_split[gpu_id]}, conv_width=0.50)
                    pretrain_path = './models/pretrained/mobilenet_v1_0.50_224_2017_06_14/mobilenet_v1_0.50_224.ckpt'
                    last_layer = 'MConv_Stage6_L{aux}_5'
                elif args.model == 'cmu':
                    net = CmuNetwork({'image': q_inp_split[gpu_id]})
                    pretrain_path = './models/numpy/openpose_coco.npy'
                    last_layer = 'Mconv7_stage6_L{aux}'
                else:
                    raise Exception('Invalid Mode.')
                vect, heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)

                l1s, l2s = net.loss_l1_l2()

                for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                    if gpu_id == 0:
                        vectmap_losses.append([])
                        heatmap_losses.append([])
                    vectmap_losses[idx].append(l1)
                    heatmap_losses[idx].append(l2)

    # define loss
    losses = []
    for l1_idx, l1 in enumerate(vectmap_losses):
        l1_concat = tf.concat(l1, axis=0)
        loss = tf.nn.l2_loss(l1_concat - q_vect, name='loss_l1_stage%d' % l1_idx)
        losses.append(loss)
    for l2_idx, l2 in enumerate(heatmap_losses):
        l2_concat = tf.concat(l2, axis=0)
        loss = tf.nn.l2_loss(l2_concat - q_heat, name='loss_l2_stage%d' % l2_idx)
        losses.append(loss)

    output_vectmap = tf.concat(output_vectmap, axis=0)
    output_heatmap = tf.concat(output_heatmap, axis=0)
    total_loss = tf.reduce_mean(losses)
    total_loss_ll_paf = tf.reduce_mean(tf.nn.l2_loss(output_vectmap - q_vect))
    total_loss_ll_heat = tf.reduce_mean(tf.nn.l2_loss(output_heatmap - q_heat))
    total_ll_loss = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])

    # define optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = args.lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decay_steps=20000, decay_rate=0.333, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_ll_loss)
    tf.summary.scalar("loss_lastlayer_paf", tf.nn.l2_loss(output_vectmap - q_vect))
    tf.summary.scalar("loss_lastlayer_heat", tf.nn.l2_loss(output_heatmap - q_heat))
    tf.summary.scalar("queue_size", enqueuer.size())
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(1, 640, 640, 3))
    valid_img = tf.summary.image('training sample', sample_train, 1)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([valid_img, valid_loss_t, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if args.checkpoint:
            logging.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logging.info('Restore from checkpoint...Done')
        elif pretrain_path:
            logging.info('Restore pretrained weights...')
            if '.ckpt' in pretrain_path:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                net.load(pretrain_path, sess, False)
            logging.info('Restore pretrained weights...Done')

        logging.info('prepare file writer')
        training_name = '{}_{}_batch:{}_lr:{}_gpus:{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            args.model,
            args.batchsize,
            args.lr,
            args.gpus
        )
        file_writer = tf.summary.FileWriter('/root/tensorboard-openpose/{}/'.format(training_name), sess.graph)

        logging.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logging.info('examine timeline')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run([train_op, global_step])
        _, gs_num = sess.run([train_op, global_step], options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

        logging.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        step_per_epoch = 121745 // args.batchsize
        initial_gs_num = sess.run(global_step)

        while True:
            _, gs_num = sess.run([train_op, global_step])

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num == initial_gs_num + 1 or gs_num - last_gs_num >= 100:
                train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary, queue_size = sess.run([total_loss, total_ll_loss, total_loss_ll_paf, total_loss_ll_heat, learning_rate, merged_summary_op, enqueuer.size()])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logging.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g, q=%d' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, queue_size))
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num == initial_gs_num + 1 or gs_num - last_gs_num2 >= 1000:
                average_loss = average_loss_ll = 0
                total_cnt = 0

                # log of test accuracy
                for images_test, heatmaps, vectmaps in validation_cache:
                    lss, lss_ll, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, total_ll_loss, output_vectmap, output_heatmap],
                        feed_dict={input_node: images_test, vectmap_node: vectmaps, heatmap_node: heatmaps}
                    )
                    average_loss += lss * len(images_test)
                    average_loss_ll += lss_ll * len(images_test)
                    total_cnt += len(images_test)

                logging.info('validation(%d) loss=%f, loss_ll=%f' % (total_cnt, average_loss / total_cnt, average_loss_ll / total_cnt))
                last_gs_num2 = gs_num

                pafMat, heatMat = sess.run(
                    [
                        net.get_output(name=last_layer.format(aux=1)),
                        net.get_output(name=last_layer.format(aux=2))
                    ], feed_dict={'image:0': np.array([val_image]*args.batchsize)}
                )
                test_result = CocoPoseLMDB.display_image(val_image, heatMat[0], pafMat[0], as_numpy=True)
                test_result = cv2.resize(test_result, (640, 640))
                test_result = test_result.reshape([1, 640, 640, 3]).astype(float)

                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss: average_loss / total_cnt,
                    valid_loss_ll: average_loss_ll / total_cnt,
                    sample_train: test_result
                })
                file_writer.add_summary(summary, gs_num)

            if gs_num > 0 and gs_num % 2000 == 0:
                saver.save(sess, os.path.join(args.modelpath, 'model'), global_step=global_step)

        saver.save(sess, os.path.join(args.modelpath, 'model_final'), global_step=global_step)
    logging.info('optimization finished. %f' % (time.time() - time_started))

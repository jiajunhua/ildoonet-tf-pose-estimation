#!/usr/bin/env python
import time
import os
import sys

import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from tfpose_ros.msg import Persons, Person, BodyPartElm

from estimator import TfPoseEstimator
from networks import model_wh, get_graph_path


def humans_to_msg(humans):
    persons = Persons()

    for human in humans:
        person = Person()

        for k in human.body_parts:
            body_part = human.body_parts[k]

            body_part_msg = BodyPartElm()
            body_part_msg.part_id = body_part.part_idx
            body_part_msg.x = body_part.x
            body_part_msg.y = body_part.y
            body_part_msg.confidence = body_part.score
            person.body_part.append(body_part_msg)
        persons.persons.append(person)

    return persons


def callback_image(data):
    # et = time.time()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr('[ros-video-recorder][VideoFrames] Converting Image Error. ' + str(e))
        return

    humans = pose_estimator.inference(cv_image)

    msg = humans_to_msg(humans)
    msg.image_w = data.width
    msg.image_h = data.height
    msg.header = data.header

    pub_pose.publish(msg)
    # rospy.loginfo(time.time() - et)


if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS', anonymous=True)

    # parameters
    image_topic = rospy.get_param('~camera', '')
    model = rospy.get_param('~model', 'cmu_640x480')

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided.')
        sys.exit(-1)

    try:
        w, h = model_wh(model)
        graph_path = get_graph_path(model)

        rospack = rospkg.RosPack()
        graph_path = os.path.join(rospack.get_path('tfpose_ros'), graph_path)
    except Exception as e:
        rospy.logerr('invalid model: %s, e=%s' % (model, e))
        sys.exit(-1)

    pose_estimator = TfPoseEstimator(graph_path, target_size=(w, h))
    cv_bridge = CvBridge()

    rospy.Subscriber(image_topic, Image, callback_image, queue_size=1)
    pub_pose = rospy.Publisher('~pose', Persons, queue_size=1)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')

from pose_dataset import CocoPose
from tensorpack import imgaug
from tensorpack.dataflow.common import MapDataComponent, MapData
from tensorpack.dataflow.image import AugmentImageComponent

from pose_augment import *


def get_idx_hands_up():
    from src.pose_augment import set_network_input_wh
    set_network_input_wh(368, 368)

    show_sample = True
    db = CocoPoseLMDB('/data/public/rw/coco-pose-estimation-lmdb/', is_train=True, decode_img=show_sample)
    db.reset_state()
    total_cnt = 0
    handup_cnt = 0
    for idx, metas in enumerate(db.get_data()):
        meta = metas[0]
        if len(meta.joint_list) <= 0:
            continue
        body = meta.joint_list[0]
        if body[CocoPart.Neck.value][1] <= 0:
            continue
        if body[CocoPart.LWrist.value][1] <= 0:
            continue
        if body[CocoPart.RWrist.value][1] <= 0:
            continue

        if body[CocoPart.Neck.value][1] > body[CocoPart.LWrist.value][1] or body[CocoPart.Neck.value][1] > body[CocoPart.RWrist.value][1]:
            print(meta.idx)
            handup_cnt += 1

            if show_sample:
                l1, l2, l3 = pose_to_img(metas)
                CocoPose.display_image(l1, l2, l3)

        total_cnt += 1

    print('%d / %d' % (handup_cnt, total_cnt))


def sample_augmentations():
    ds = CocoPose('/data/public/rw/coco-pose-estimation-lmdb/', is_train=False, only_idx=0)
    ds = MapDataComponent(ds, pose_random_scale)
    ds = MapDataComponent(ds, pose_rotation)
    ds = MapDataComponent(ds, pose_flip)
    ds = MapDataComponent(ds, pose_resize_shortestedge_random)
    ds = MapDataComponent(ds, pose_crop_random)
    ds = MapData(ds, pose_to_img)
    augs = [
        imgaug.RandomApplyAug(imgaug.RandomChooseAug([
            imgaug.GaussianBlur(3),
            imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
            imgaug.RandomOrderAug([
                imgaug.BrightnessScale((0.8, 1.2), clip=False),
                imgaug.Contrast((0.8, 1.2), clip=False),
                # imgaug.Saturation(0.4, rgb=True),
            ]),
        ]), 0.7),
    ]
    ds = AugmentImageComponent(ds, augs)

    ds.reset_state()
    for l1, l2, l3 in ds.get_data():
        CocoPose.display_image(l1, l2, l3)


if __name__ == '__main__':
    # codes for tests
    # get_idx_hands_up()

    # show augmentation samples
    sample_augmentations()

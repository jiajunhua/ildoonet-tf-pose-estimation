from common import CocoPart
from pose_augment import pose_to_img
from pose_dataset import CocoPoseLMDB


def get_idx_hands_up():
    from pose_augment import set_network_input_wh
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
                CocoPoseLMDB.display_image(l1, l2, l3)

        total_cnt += 1

    print('%d / %d' % (handup_cnt, total_cnt))


if __name__ == '__main__':
    get_idx_hands_up()

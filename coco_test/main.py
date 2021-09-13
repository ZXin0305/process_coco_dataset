from pycocotools.coco import COCO  #导入coco
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from IPython import embed
import json
import numpy as np
from path import Path
import cv2

root_dir = '/media/zx/新加卷/coco2017/annotations'
data_type = 'train2017'
anno_file = os.path.join(root_dir, 'person_keypoints_{}.json'.format(data_type))
coco_kps = COCO(anno_file)
json_file_dirs = Path('/media/xuchengjun/datasets/COCO')

catIds = coco_kps.getCatIds(catNms=['person'])
imgIds = coco_kps.getImgIds(catIds=catIds)

COCO2CMUF = [-1,0,-1,5,7,9,11,13,15,6,8,10,12,13,14]

def draw(body,img):
    for i in range(17):
        cv2.circle(img, center=(body[i][0],body[i][1]), radius=4, color=(0,0,255))
        cv2.imshow('1',img)
        cv2.waitKey(0)

def main():
    count = 0

    for i in range(len(imgIds)):
        output_json = dict()
        img = coco_kps.loadImgs(imgIds[i])[0]  # [{...}]
        pixel_coors = []

        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
        annos = coco_kps.loadAnns(annIds)  #list all person keypoints in the img
        for anno in annos:
            if anno['num_keypoints'] < 3:
                continue
            body = np.asarray(anno['keypoints'])
            embed()
            body.resize((17,3))
            # img = cv2.imread(img_path)
            # draw(body, img)
            # body_tmp = np.zeros((3,17))
            # body_tmp[0] = body[:,0]
            # body_tmp[1] = body[:,1]
            # body_tmp[2] = body[:,2]

            body_new = np.zeros((15,4))

            for k in range(len(COCO2CMUF)):
                if COCO2CMUF[k] < 0:
                    continue
                body_new[k][0] = body[COCO2CMUF[k]][0]
                body_new[k][1] = body[COCO2CMUF[k]][1]
                body_new[k][2] = 0

            middle_shoulder = (body[5] + body[6]) / 2
            middle_hip = (body[11] + body[12]) / 2

            #hip 
            body_new[2][0] = middle_hip[0]
            body_new[2][1] = middle_hip[1]
            body_new[2][2] = 0    #no depth

            #neck
            body_new[0][0] = (middle_shoulder[0] - middle_hip[0])*0.185 + middle_shoulder[0]
            body_new[0][1] = (middle_shoulder[1] - middle_hip[1])*0.185 + middle_shoulder[1]
            body_new[0][2] = 0     

            body_tmp = np.zeros(shape=(3,15))
            body_tmp[0] = body_new[:,0] 
            body_tmp[1] = body_new[:,1] 
            body_tmp[2] = body_new[:,2] 
            pixel_coors.append(body_tmp.tolist())

        if len(pixel_coors) < 1:
            continue 
        img_root = Path('/media/xuchengjun/datasets/coco_2017') / data_type
        img_path = img_root / img['file_name']
        output_json['img_path'] = img_path
        output_json['pixel_coors'] = pixel_coors

        fx = img['width'] 
        fy = img['width']
        cx = img['width'] / 2 
        cy = img['height'] / 2
        cam = np.zeros(shape=(3,3))
        cam[0,0], cam[0,2] = fx, cx
        cam[1,1], cam[1,2] = fy, cy
        cam[2,2] = 1

        output_json['cam'] = cam.tolist()
        output_json['img_width'] = img['width']
        output_json['img_height'] = img['height']

        output_json['img_id'] = img['id']
        output_json['cam_id'] = 0

        img_id = img['id']
        output_json_path = json_file_dirs / f'{img_id}.json'
        with open(output_json_path, 'w') as f:
            json.dump(output_json, f)

        count += 1
        print(f'working .. {count}')
    
    print(f'has done {count}')


if __name__ == '__main__':
    main()

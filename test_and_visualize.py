import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

def main():
    config_frcnn = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
    checkpoint_frcnn = '/content/work_dirs/faster_rcnn_voc/latest.pth'

    config_yolo = '/content/mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py'
    checkpoint_yolo = '/content/work_dirs/yolov3_voc/latest.pth'

    model_frcnn = init_detector(config_frcnn, checkpoint_frcnn, device='cuda:0')
    model_yolo = init_detector(config_yolo, checkpoint_yolo, device='cuda:0')

    img_paths = [
        '/content/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
        '/content/data/VOCdevkit/VOC2007/JPEGImages/000002.jpg',
        '/content/data/VOCdevkit/VOC2007/JPEGImages/000003.jpg',
        '/content/data/VOCdevkit/VOC2007/JPEGImages/000004.jpg'
    ]

    for img_path in img_paths:
        result_frcnn = inference_detector(model_frcnn, img_path)
        show_result_pyplot(model_frcnn, img_path, result_frcnn, title='Faster R-CNN')

        result_yolo = inference_detector(model_yolo, img_path)
        show_result_pyplot(model_yolo, img_path, result_yolo, title='YOLO V3')

    new_img_paths = [
        '/content/new_image1.jpg',
        '/content/new_image2.jpg',
        '/content/new_image3.jpg'
    ]

    for img_path in new_img_paths:
        result_frcnn = inference_detector(model_frcnn, img_path)
        show_result_pyplot(model_frcnn, img_path, result_frcnn, title='Faster R-CNN')

        result_yolo = inference_detector(model_yolo, img_path)
        show_result_pyplot(model_yolo, img_path, result_yolo, title='YOLO V3')

if __name__ == '__main__':
    main()


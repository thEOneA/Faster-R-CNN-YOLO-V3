import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed

def main():
    cfg = Config.fromfile('/content/mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py')
    cfg.dataset_type = 'VOCDataset'
    cfg.data_root = '/content/data/VOCdevkit/'

    cfg.data.test.ann_file = '/content/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    cfg.data.train.ann_file = [
        '/content/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        '/content/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
    ]
    cfg.data.val.ann_file = '/content/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 2
    cfg.optimizer.lr = 0.001
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10
    cfg.total_epochs = 12
    cfg.work_dir = '/content/work_dirs/yolov3_voc'
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    os.makedirs(cfg.work_dir, exist_ok=True)
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES

    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()

'''
!pip install -U torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu121/torch_stable.html
!pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch1.13.1/index.html
!pip install mmdet==2.17.0
!pip install matplotlib opencv-python
!pip install mmengine==0.7.1


!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -r requirements/build.txt
!pip install -v -e .


!mkdir -p /content/data/VOCdevkit
%cd /content/data/VOCdevkit
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf VOCtrainval_06-Nov-2007.tar
!tar -xvf VOCtest_06-Nov-2007.tar
!tar -xvf VOCtrainval_11-May-2012.tar
%cd /content
'''

# YOLOv7 with Oriented Bounding Box
This respository employs the YOLOv7 as the backbone and work with KFIoU, KLD, and CSL in the loss function.

## Credits
We obtain some codes from the following authors.

* <a href='https://github.com/WongKinYiu/yolov7'>YOLOv7</a>
* <a href='https://github.com/SSTato/YOLOv7_obb'>YOLOv7-CSL</a>
* <a href='https://github.com/lx-cly/YOLOv7_OBB'>YOLOv7-KLD</a>
* <a href='https://github.com/open-mmlab/mmrotate/blob/6519a3654e17b707c15d4aa2c5db1257587ea4c0/mmrotate/models/losses/kf_iou_loss.py'>KFIoU in MMRotate</a>
* <a href='https://github.com/hukaixuan19970627/yolov5_obb'>YOLOv5-CSL</a>

## Quick Start

```
$ pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
$ nvcc -V
$ nvidia-smi

$ git clone https://github.com/worldstar/yolov7-obb.git
$ cd yolov7-obb/
$ pip install -r requirements.txt

$ cd utils/nms_rotated
$ python setup.py develop  #or "pip install -v -e ."

$ cd ../
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
$ python train.py --data ./dataset/Aerial-Airport-1/data.yaml --mode KLD --weights yolov7.pt --epochs 20 --batch-size 8 --img 1024 --device 0 --exist-ok
```

## Datasets
* <a href='https://captain-whu.github.io/DOTA/dataset.html'>DOTA</a>
* <a href='https://paperswithcode.com/dataset/dota'>Paper with code</a>

## Labeling tool
* <a href='https://rectlabel.com/'>RectLabel</a>
* <a href='https://labelbox.com/'>LabelBox</a>
* <a href='https://github.com/heartexlabs/labelImg'>LabelImg</a>
* <a href='https://public.roboflow.com/object-detection'>Roboflow</a>

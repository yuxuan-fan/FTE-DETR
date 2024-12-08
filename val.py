import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'E:\githubFYX\RTDETR-main\runs\train\ASFP2-Faster P2200\weights\last.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=4,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp图像',
              )
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'E:\githubFYX\RTDETR-main\ultralytics\cfg\models\rt-detr\rtdetr-r18.yaml')
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=350,
                batch=6,
                workers=4,
                
                device='0',
                # device='cpu', #本地调试使用cpu
                # resume='', # last.pt path
                project='runs/train',
                name='deepPCB',
                )

    

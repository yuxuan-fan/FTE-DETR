
# 实验环境配置:
python: 3.8.16
torch: 1.13.1+cu117
torchvision: 0.14.1+cu117
timm: 0.9.8
mmcv: 2.1.0
mmengine: 0.9.0


```
conda create --name rtdetr python=3.8.16
pip安装包时，可能需要关闭vpn
conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch
pip install timm==0.9.8
pip install mmcv==2.1.0   #这个setup可能有点问题，直接中止不影响
pip install mmengine==0.9.0

pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple

```







# 数据集相关
VOC格式的数据集将标注放在Annotations,图片放在JPEGImages，图片后缀需要固定。
.\dataset\VOCdevkit\Annotations

运行xml2txt.py，会将xml格式的标注转换成txt，需注意路径。
这样就得到了yolo格式的数据集。

通过split_data.py对数据集进行划分，运行完就会分割出iamges和lables文件夹

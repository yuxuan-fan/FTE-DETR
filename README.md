无预训练权重，需下载


本项目使用的ultralytics版本为8.0.201,在ultralytics/__init__.py中的__version__有标识.


# 实验环境配置:
用anaconda创建虚拟环境，

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

```


```
常规步骤
1. 执行pip uninstall ultralytics把安装在环境里面的ultralytics库卸载干净.
2. 卸载完成后同样再执行一次,如果出现WARNING: Skipping ultralytics as it is not installed.证明已经卸载干净.
3. 如果需要使用官方的CLI运行方式,需要把ultralytics库安装一下,执行命令:<python setup.py develop>,当然安装后对本代码进行修改依然有效.(develop作用解释具体可看: https://blog.csdn.net/qq_16568205/article/details/110433714)  注意:不需要使用官方的CLI运行方式,可以选择跳过这步


4. 额外需要的包安装命令:
    pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple


    以下主要是使用dyhead必定需要安装的包,如果安装不成功dyhead没办法正常使用!
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"

5. 运行时候如果还缺什么包就请自行安装即可.（后面的改进模块可能需要引入新的包）

需要编译才能运行的一些模块:
    1. mamba(百度云视频-20240219更新说明)
    2. dcnv3(百度云视频-20231119更新说明)
    3. dcnv4(百度云视频-20240120更新说明)
    4. smpconv(百度云视频-20240608更新说明)
    5. mamba-yolo(百度云视频-20240622更新说明)

本目录下的test_env.py文件为了验证一些需要编译的或者难安装的(mmcv)是否成功的代码.详细请看以下这期视频:https://pan.baidu.com/s/1sWwvN4UC3blBRVe1twrJAg?pwd=bru5
```





# 数据集相关
VOC格式的数据集将标注放在Annotations,图片放在JPEGImages，图片后缀需要固定。
.\dataset\VOCdevkit\Annotations

运行xml2txt.py，会将xml格式的标注转换成txt，需注意路径。
这样就得到了yolo格式的数据集。

通过split_data.py对数据集进行划分，运行完就会分割出iamges和lables文件夹

data.yaml中需要修改项目路径 类别数量 类别名称


# 自带的一些文件说明
1. train.py
    训练模型的脚本 导入预训练权重
2. main_profile.py
    输出模型和模型每一层的参数,计算量的脚本(rtdetr-l和rtdetr-x因为thop库的问题,没办法正常输出每一层的参数和计算量和时间)
3. val.py
    使用训练好的模型计算指标的脚本
4. detect.py
    推理的脚本
5. track.py
    跟踪推理的脚本
6. heatmap.py
    生成热力图的脚本
7. get_FPS.py
    计算模型储存大小、模型推理时间、FPS的脚本
8. get_COCO_metrice.py
    计算COCO指标的脚本
9. plot_result.py
    绘制曲线对比图的脚本
10. get_model_erf.py
    绘制模型的有效感受野.[视频链接](https://www.bilibili.com/video/BV1Gx4y1v7ZZ/)
11. export.py
    导出模型脚本

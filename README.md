无预训练权重，需下载
推荐用r18，做轻量化


本项目使用的ultralytics版本为8.0.201,在ultralytics/__init__.py中的__version__有标识.

实验环境:

python: 3.8.16

torch: 1.13.1+cu117

torchvision: 0.14.1+cu117

timm: 0.9.8

mmcv: 2.1.0

mmengine: 0.9.0


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



# 数据集相关
VOC格式的数据集将标注放在Annotations,图片放在JPEGImages，图片后缀需要固定。
.\dataset\VOCdevkit\Annotations

运行xml2txt.py，会将xml格式的标注转换成txt，需注意路径。
这样就得到了yolo格式的数据集。

通过split_data.py对数据集进行划分，运行完就会分割出iamges和lables文件夹

data.yaml中需要修改项目路径 类别数量 类别名称

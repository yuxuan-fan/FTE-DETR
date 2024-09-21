from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    Attributes:
        预训练模型的路径
        model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.
    """

    def __init__(self, model='rtdetr-l.pt') -> None:
        """
        Initializes the RT-DETR model with the given pre-trained model file. Supports .pt and .yaml formats.
        支持 .pt 和 .yaml 格式。

        Args:
            model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
            如果 model 不为空且 model 字符串以点分割后的最后一部分（文件扩展名），不在 ('pt', 'yaml', 'yml') 这个元组中
            抛出一个未实现错误
        """
        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt, *.yaml, or *.yml files.')
        super().__init__(model=model, task='detect')# 调用父类的初始化方法，传入 model 和 task 参数，其中 task 为'detect'

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer,
                'model': RTDETRDetectionModel}}

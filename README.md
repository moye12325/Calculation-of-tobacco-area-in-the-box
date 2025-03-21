## 版本说明
- V4版本：最后几个小版本主要是更新batchsize，imagesize
- V5版本：
  - V5.0： 只**加入随机翻转、旋转**以实现数据增强。训练集使用数据增强，训练集不使用。reRun了一遍，效果不错，但与V4相比效果未知。
  - V5.1： 在5.0的基础上**加入裁剪**，新增"crop_size"参数，表示裁剪为目标尺寸的比例。
  - V5.2： reRun5.1，发现前两次训练有问题，推理图片全黑。reRun后正常，但效果差，裁剪还是不能加。
  - V5.3： 修改train，不加入裁剪，保留my_dataset.py里面的裁剪功能。
  - V5.4： **加入albumentations进行数据增强**。详细功能见注释。

## TODO
- 确定后续训练的图像尺寸
- 将推理后的优质图片作为mask当作训练样本继续训练
- 标注一些絮状复杂的图片作为训练样本
- **加入Warmup学习率预热**，前期使用较小的学习率逐步增加，再进入正常训练阶段，避免一开始较大的梯度更新造成的不稳定
- **引入IoU交并比**，即为预测区域和真实标注区域的重叠部分除以它们的联合部分
- 引入Dice系数
### 环境初始化
安装依赖

    python 3.6

    pip install -U -r requirements.txt


### 训练

    python train.py --dataset_root ~/ai/data --epochs 30 --cuda 1

- voc_root 训练集路径，包含 Image(图片)和Annotation(标注)两个目录
- epochs 循环次数
- cuda 表示使用gpu


初始加载`weights/vgg16_reducedfc.pth`开始训练。


### 验证 -- 计算mAP

    python eval.py --trained_model weights/gpu.pth --voc_root ~/ai/data/test --cuda=1


### 测试结果

    python test.py --trained_model weights/gpu.pth --voc_root ~/ai/data/test --cuda=1


把预测出的位置画在图上，并保存在 ImageTarget目录中

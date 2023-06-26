import paddle
import cv2
import numpy as np
from PIL import Image
label_dic = {'0': 'dryer', '1': 'power', '2': 'watch'}



# 定义卷积池化网络
class ConvPool(paddle.nn.Layer):
    '''卷积+池化'''

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 groups,
                 conv_stride=1,
                 conv_padding=1,
                 ):
        super(ConvPool, self).__init__()

        # groups代表卷积层的数量
        for i in range(groups):
            self.add_sublayer(  # 添加子层实例
                'bb_%d' % i,
                paddle.nn.Conv2D(  # layer
                    in_channels=num_channels,  # 通道数
                    out_channels=num_filters,  # 卷积核个数
                    kernel_size=filter_size,  # 卷积核大小
                    stride=conv_stride,  # 步长
                    padding=conv_padding,  # padding
                )
            )
            self.add_sublayer(
                'relu%d' % i,
                paddle.nn.ReLU()
            )
            num_channels = num_filters

        self.add_sublayer(
            'Maxpool',
            paddle.nn.MaxPool2D(
                kernel_size=pool_size,  # 池化核大小
                stride=pool_stride  # 池化步长
            )
        )

    def forward(self, inputs):
        x = inputs
        for prefix, sub_layer in self.named_children():
            # print(prefix,sub_layer)
            x = sub_layer(x)
        return x


# VGG网络
class VGGNet(paddle.nn.Layer):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 5个卷积池化操作
        self.convpool01 = ConvPool(
            3, 64, 3, 2, 2, 2)  # 3:通道数，64：卷积核个数，3:卷积核大小，2:池化核大小，2:池化步长，2:连续卷积个数
        self.convpool02 = ConvPool(
            64, 128, 3, 2, 2, 2)
        self.convpool03 = ConvPool(
            128, 256, 3, 2, 2, 3)
        self.convpool04 = ConvPool(
            256, 512, 3, 2, 2, 3)
        self.convpool05 = ConvPool(
            512, 512, 3, 2, 2, 3)
        self.pool_5_shape = 512 * 7 * 7
        # 三个全连接层
        self.fc01 = paddle.nn.Linear(self.pool_5_shape, 4096)
        self.drop1 = paddle.nn.Dropout(p=0.5)
        self.fc02 = paddle.nn.Linear(4096, 4096)
        self.drop2 = paddle.nn.Dropout(p=0.5)
        self.fc03 = paddle.nn.Linear(4096, 3)

    def forward(self, inputs, label=None):
        # print('input_shape:', inputs.shape) #[8, 3, 224, 224]
        """前向计算"""
        out = self.convpool01(inputs)
        # print('convpool01_shape:', out.shape)           #[8, 64, 112, 112]
        out = self.convpool02(out)
        # print('convpool02_shape:', out.shape)           #[8, 128, 56, 56]
        out = self.convpool03(out)
        # print('convpool03_shape:', out.shape)           #[8, 256, 28, 28]
        out = self.convpool04(out)
        # print('convpool04_shape:', out.shape)           #[8, 512, 14, 14]
        out = self.convpool05(out)
        # print('convpool05_shape:', out.shape)           #[8, 512, 7, 7]

        out = paddle.reshape(out, shape=[-1, 512 * 7 * 7])
        out = self.fc01(out)
        out = self.drop1(out)
        out = self.fc02(out)
        out = self.drop2(out)
        out = self.fc03(out)

        if label is not None:
            acc = paddle.metric.accuracy(input=out, label=label)
            return out, acc
        else:
            return out



def load_image(img_path="./test.jpg"):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path)
    # img = cv2.imread(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1)) / 255  # HWC to CHW 及归一化
    return img



print(1)
model__state_dict = paddle.load('./model/save_dir_final.pdparams')
model_predict = VGGNet()
model_predict.set_state_dict(model__state_dict)
model_predict.eval()
print(2)
infer_img = load_image("test.jpg")
print(3)
# infer_img = infer_img[np.newaxis, :, :, :]  # reshape(-1,3,224,224)
infer_img = np.array([infer_img])
print(3.5)
infer_img = paddle.to_tensor(infer_img)
print(4)
result = model_predict(infer_img)
print(5)
lab = np.argmax(result.numpy())
text = "样本预测结果为：{}".format(label_dic[str(lab)])
print("样本: {},被预测为:{}".format("test.jpg", label_dic[str(lab)]))
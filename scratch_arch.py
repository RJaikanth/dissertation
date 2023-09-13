import torch
from torch.utils.tensorboard.writer import SummaryWriter

# from ultralytics.nn import modules as nn
from torchinfo import summary
import torch.nn as nn
from ultralytics.nn.tasks import parse_model, yaml_model_load
from ultralytics.nn.modules import block, conv, head

from pprint import pprint


class YOLOposeXModel(nn.Module):
    def __init__(self, scale='x'):
        super().__init__()

        ch = 3
        d = yaml_model_load("./yolov8-pose.yaml")
        d['scale'] = scale

        self.model, self.save = parse_model(d, 3, False)
        m = self.model[-1]
        s = 256
        m.inplace = True
        forward = lambda x: self.forward(x)[0]
        m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
        self.stride = m.stride

    def forward(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def predict(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x


class YOLOposeNModel(nn.Module):
    def __init__(self, scale='n'):
        super().__init__()

        ch = 3
        d = yaml_model_load("./yolov8-pose.yaml")
        d['scale'] = scale

        self.model, self.save = parse_model(d, 3, False)
        m = self.model[-1]
        s = 256
        m.inplace = True
        forward = lambda x: self.forward(x)[0]
        m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
        self.stride = m.stride

    def forward(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def predict(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat([], self.d)


class Pose(nn.Module):
    def __init__(self):
        super().__init__()

        ch = 3
        d = yaml_model_load("./yolov8-pose.yaml")
        d['scale'] = 'n'

        self.model, self.save = parse_model(d, 3, False)
        self.model = self.model[22]
        # print(self.model[22])

    def forward(self, x):
        x = [torch.zeros((1, 64, 80, 80)), torch.zeros((1, 128, 40, 40)), torch.zeros((1, 256, 20, 20))]
        return self.model(x)


if __name__ == "__main__":
    # modelx = YOLOposeXModel()
    # summary(modelx, input_size = (1, 3, 640, 640), col_names = col_names)
    # writer = SummaryWriter("./arch/yolo8x", comment = "yolo8x")
    # writer.add_graph(modelx, torch.zeros(1, 3, 640, 640))
    # writer.flush()
    # writer.close()

    # modeln = YOLOposeNModel()
    # summary(modeln, input_size = (1, 3, 640, 640), col_names = col_names)
    # writer = SummaryWriter("./arch/yolo8n", comment = "yolo8n")
    # writer.add_graph(modeln, torch.zeros(1, 3, 640, 640))
    # writer.flush()
    # writer.close()

    # model = block.Conv(3, 16, 3, 2)
    # writer = SummaryWriter("./arch/conv", comment = "yolo8x")
    # writer.add_graph(model, torch.zeros(1, 3, 640, 640))
    # writer.flush()
    # writer.close()

    # model = block.Bottleneck(16, 16, True, 1, k = ((3, 3), (3, 3)), e = 1.0)
    # writer = SummaryWriter("./arch/bottleneck", comment = "yolo8x")
    # writer.add_graph(model, torch.zeros(1, 16, 160, 160))
    # writer.flush()
    # writer.close()

    # model = block.C2f(64, 64, 2, True)
    # writer = SummaryWriter("./arch/c2f", comment = "yolo8x")
    # writer.add_graph(model, torch.zeros(1, 64, 80, 80))
    # writer.flush()
    # writer.close()

    # model = block.SPPF(256, 256, 5)
    # writer = SummaryWriter("./arch/sppf", comment = "yolo8x")
    # writer.add_graph(model, torch.zeros(1, 256, 20, 20))
    # writer.flush()
    # writer.close()

    # model = block.SPPF(256, 256, 5)
    # writer = SummaryWriter("./arch/sppf", comment = "yolo8x")
    # writer.add_graph(model, torch.zeros(1, 256, 20, 20))
    # writer.flush()
    # writer.close()

    # model = Concat(1)
    # writer = SummaryWriter("./arch/concat", comment = "yolo8x")
    # writer.add_graph(model, input_to_model=list((torch.zeros(1, 128, 20, 20), torch.zeros(1, 256, 20, 20))))
    # writer.flush()
    # writer.close()

    # model = head.Pose(1, [17, 3], [64, 128, 256])
    # writer = SummaryWriter("./arch/posehead", comment = "yolo8x")
    # writer.add_graph(model, input_to_model = list((torch.zeros(1, 128, 20, 20), torch.zeros(1, 256, 20, 20))))
    # writer.flush()
    # writer.close()

    # from ultralytics import YOLO
    # x = torch.zeros((1, 3, 640, 640))
    # model = YOLO("./yolov8-pose.yaml")
    # model.info(False)
    # summary(model, input_data = x, depth=4)
    # writer = SummaryWriter("./arch/pose", comment = "yolo8n")
    # writer.add_graph(model, torch.zeros(1, 3, 640, 640))
    # writer.flush()
    # writer.close()

    # 15 [1, 64, 80, 80]
    # 18 [1, 128, 40, 40]
    # 21

    # 0 <class 'ultralytics.nn.modules.conv.Conv'> 3 16[3, 16, 3, 2]
    # 1 <class 'ultralytics.nn.modules.conv.Conv'> 16 32[16, 32, 3, 2]
    # 2 <class 'ultralytics.nn.modules.block.C2f'> 32 32[32, 32, 1, True]
    # 3 <class 'ultralytics.nn.modules.conv.Conv'> 32 64[32, 64, 3, 2]
    # 4 <class 'ultralytics.nn.modules.block.C2f'> 64 64[64, 64, 2, True]
    # 5 <class 'ultralytics.nn.modules.conv.Conv'> 64 128[64, 128, 3, 2]
    # 6 <class 'ultralytics.nn.modules.block.C2f'> 128 128[128, 128, 2, True]
    # 7 <class 'ultralytics.nn.modules.conv.Conv'> 128 256[128, 256, 3, 2]
    # 8 <class 'ultralytics.nn.modules.block.C2f'> 256 256[256, 256, 1, True]
    # 9 <class 'ultralytics.nn.modules.block.SPPF'> 256 256[256, 256, 5]
    # 10 <class 'torch.nn.modules.upsampling.Upsample'> 256 256[None, 2, 'nearest']
    # 11 <class 'ultralytics.nn.modules.conv.Concat'> 256 384[1]
    # 12 <class 'ultralytics.nn.modules.block.C2f'> 384 128[384, 128, 1]
    # 13 <class 'torch.nn.modules.upsampling.Upsample'> 384 128[None, 2, 'nearest']
    # 14 <class 'ultralytics.nn.modules.conv.Concat'> 384 192[1]
    # 15 <class 'ultralytics.nn.modules.block.C2f'> 192 64[192, 64, 1]
    # 16 <class 'ultralytics.nn.modules.conv.Conv'> 64 64[64, 64, 3, 2]
    # 17 <class 'ultralytics.nn.modules.conv.Concat'> 64 192[1]
    # 18 <class 'ultralytics.nn.modules.block.C2f'> 192 128[192, 128, 1]
    # 19 <class 'ultralytics.nn.modules.conv.Conv'> 128 128[128, 128, 3, 2]
    # 20 <class 'ultralytics.nn.modules.conv.Concat'> 128 384[1]
    # 21 <class 'ultralytics.nn.modules.block.C2f'> 384 256[384, 256, 1]
    # 22 <class 'ultralytics.nn.modules.head.Pose'> 384 256[1, [17, 3], [64, 128, 256]]

    from src.poselift.cnn import CNNModel, ResidualBlock
    from src.poselift.ffn import FFNModel

    model = ResidualBlock(16, 32)
    writer = SummaryWriter("./arch/res_block")
    writer.add_graph(model, input_to_model = torch.zeros((1, 16, 2)))
    writer.flush()
    writer.close()

    model = FFNModel()
    writer = SummaryWriter("./arch/ffn")
    writer.add_graph(model, input_to_model = torch.zeros((1, 17, 2)))
    writer.flush()
    writer.close()

    model = CNNModel()
    writer = SummaryWriter("./arch/cnn")
    writer.add_graph(model, input_to_model = torch.zeros((1, 17, 2)))
    writer.flush()
    writer.close()

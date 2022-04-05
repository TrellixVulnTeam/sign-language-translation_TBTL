import torch
import torch.nn as nn
import numpy as np

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            # conv2
            nn.Conv2d(96, 256, 5, padding=2, groups=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            # conv3
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            # conv4
            nn.Conv2d(384, 384, 3, padding=1, groups=2),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            # conv5
            nn.Conv2d(384, 256, 3, padding=1, groups=2),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x



def alexnet(num_classes=1000, pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    for name, param in model.named_parameters():
        print(name,param.shape)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    if pretrained:
        weights_dict = np.load('/nas1/yjun/slt/bvlc_alexnet.npy', encoding='bytes', allow_pickle=True).item()
        weights_key = sorted(list(weights_dict.keys()))
        conv_weights_key = [x for x in weights_key if x.startswith('conv')]
        fc_weights_key = [x for x in weights_key if x.startswith('fc')]
        # conv layer
        idx = 0
        for i in range(len(model.features)):
            if str(type(model.features[i])) == "<class 'torch.nn.modules.conv.Conv2d'>":
                model.features[i].weight = torch.nn.Parameter(torch.Tensor((weights_dict[conv_weights_key[idx]][0].T)))
                model.features[i].bias = torch.nn.Parameter(torch.Tensor(weights_dict[conv_weights_key[idx]][1]).T)
                idx += 1
        # fc layer
        idx = 0
        for i in range(len(model.classifier)):
            if str(type(model.classifier[i])) == "<class 'torch.nn.modules.linear.Linear'>":
                if i == len(model.classifier)-1:
                    nn.init.xavier_normal_(model.classifier[i].weight)
                    nn.init.zeros_(model.classifier[i].bias)
                else:
                    model.classifier[i].weight = torch.nn.Parameter(torch.Tensor(weights_dict[fc_weights_key[idx]][0]).T)
                    model.classifier[i].bias = torch.nn.Parameter(torch.Tensor(weights_dict[fc_weights_key[idx]][1]).T)
                idx += 1                                   
        for key in weights_key:
            
            if key.startswith('fc'):
                layer = f'classifier.{key[-1]}.'
            
            elif key.startswith('conv'):
                layer = f'features.{key[-1]}.'
            else:
                print('layer not in alexnet')
            for arr in weights_dict[key]:
                # Biases
                if len(arr.shape) == 1:
                    layer_key = layer + 'bias'
                else:
                    layer_key = layer + 'weight'
                model.state_dict()[layer_key] = torch.Tensor(arr)
    return model

if __name__ == "__main__":
    model = alexnet(pretrained=True)
    
    

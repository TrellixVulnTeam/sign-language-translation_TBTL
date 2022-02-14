import torch
import torch.nn as nn
from torchvision import models, transforms

import PIL
import glob
import pandas as pd
from tqdm import tqdm
import pickle
import time
import numpy as np


class AlexNet(nn.Module):
  def __init__(self, num_classes=1024):
    super().__init__()
    ##### CNN layers 
    self.net = nn.Sequential(
        # conv1
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
        nn.ReLU(inplace=True),  # non-saturating function
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # 논문의 LRN 파라미터 그대로 지정
        nn.MaxPool2d(kernel_size=3, stride=2),
        # conv2
        nn.Conv2d(96, 256, kernel_size=5, padding=2), 
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # conv3
        nn.Conv2d(256, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # conv4
        nn.Conv2d(384, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # conv5
        nn.Conv2d(384, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

    )

    ##### FC layers
    self.classifier = nn.Sequential(
        # fc1
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
        nn.ReLU(inplace=True).
        # fc2
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
    # bias, weight 초기화 
    def init_bias_weights(self):
      for layer in self.net:
        if isinstance(layer, nn.Conv2d):
          nn.init.normal_(layer.weight, mean=0, std=0.01)   # weight 초기화
          nn.init.constant_(layer.bias, 0)   # bias 초기화
      # conv 2, 4, 5는 bias 1로 초기화 
      nn.init.constant_(self.net[4].bias, 1)
      nn.init.constant_(self.net[10].bias, 1)
      nn.init.constant_(self.net[12].bias, 1)
    # modeling 
    def forward(self, x):
      x = self.net(x)   # conv
      x = x.view(-1, 256*6*6)   # keras의 reshape (텐서 크기 2d 변경)
      return self.classifier(x)   # fc   

class SpatialEmbedding(object):
    def __init__(self, model_name, num_classes, feature_extract=True, use_pretrained=True):
        """Create the graph of the Spatial Embedding model.

        Args:
            source: Placeholder for the input tensor.
            keep_prob: Dropout probability.
        """
        
        # Parse input arguments into class variables
        # self.input = source
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        
    def create_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

 
        model_ft = models.alexnet(pretrained=self.use_pretrained)
        self.set_parameter_requires_grad(model_ft, self.feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
        input_size = 227

        return model_ft, input_size
    
    def set_parameter_requires_grad(self, model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
                    

def extractor(model_name, num_classes, phase='train', gpu=0, feature_extract=True, use_pretrained=True, save_pickle=False):
    se = SpatialEmbedding(model_name, num_classes, feature_extract, use_pretrained)
    device = f'cuda:{gpu}'
    # create the model
    model_ft, input_size = se.create_model()
    model_ft = model_ft.to(device)
    
    trans = transforms.Compose([transforms.Resize((input_size)), 
                            transforms.ToTensor()])
    
    # trans = transforms.Compose([transforms.Resize((input_size)), 
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])    
    # Print the model we just instantiated
    print(model_ft)
    
    # 나중에 parser로 받을 부분
    phase = phase
    upper_dir = '/nas1/yjun/slt/PHOENIX-2014-T/'
    img_dir = upper_dir + 'features/fullFrame-210x260px/'
    annotation_dir = upper_dir +  '/annotations/manual/'
    annotation_file = annotation_dir + f'PHOENIX-2014-T.{phase}.corpus.csv'
    annotation = pd.read_csv(annotation_file, sep='|')[['name', 'speaker', 'orth', 'translation']]
    
    dataset = []
    for n in tqdm(range(len(annotation)), desc="SpEmbd Seq"):
        name, speaker, orth, translation = annotation.iloc[n, :]
        img_list = glob.glob(img_dir + phase + '/' + name + '/*.png')
        init = True
        for i in img_list:
            img = PIL.Image.open(i)
            input_tensor = trans(img)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            output = model_ft(input_tensor)
            output = output.cpu()
            output = output.detach().numpy()
            if init:
                seq_tensor = output
                init = False
                continue
            seq_tensor = np.vstack([seq_tensor, output])
        seq_tensor = torch.from_numpy(seq_tensor)
        seq_dic = {
            'name' : phase + '/' + name,
            'signer' : speaker,
            'gloss' : orth,
            'text' : translation,
            'sign' : seq_tensor
        }
        dataset.append(seq_dic)
    now = time.strftime("%Y%m%d-%H%M%S")
    with open(phase + f'{now}.pkl', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset

if __name__ == "__main__":
    for p in ['train', 'dev', 'test']:
        extractor('alexnet', 1024, phase=p)

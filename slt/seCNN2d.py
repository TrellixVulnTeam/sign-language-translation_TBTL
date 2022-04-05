from pyexpat import model
import torch
import torch.nn as nn
from torchvision import models, transforms
from alexnet_custom import alexnet

import PIL
import glob
import pandas as pd
from tqdm import tqdm
import pickle
import time
import numpy as np


class SpatialEmbedding():
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

        if self.model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            # model_ft = models.alexnet(pretrained=self.use_pretrained)
            model_ft = alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
            input_size = (227, 227)

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
            input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,self.num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()
        print('parameter tensor')
        for param_tensor in model_ft.state_dict():
            print(param_tensor, "\t", model_ft.state_dict()[param_tensor].size())
        # print(model_ft.state_dict()['features.0.weight'])
        print('-'*100, '\n', '\n')
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
    print('model print: ', model_ft, '\n', '-'*100, '\n', '\n')
    
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
    for p in ['train','dev', 'test']:
        extractor('alexnet', 1024, phase=p)

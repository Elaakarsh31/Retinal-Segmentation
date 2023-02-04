import torch
import torch.nn as nn

class A:
    def __init__(self):
        print("Hello")

class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_c)

    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_c)

    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    
    return x

class encoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.conv = conv_block(in_c, out_c)
    self.pool = nn.MaxPool2d((2,2))

  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)

    return x, p

class decoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    self.conv = conv_block(out_c + out_c, out_c)

  def forward(self, inputs, skip):
    x = self.up(inputs)
    x = torch.cat([x, skip], axis=1)
    x = self.conv(x)
    return x

import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_PATH = "data/image.png"

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    # mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

class Res34Unet(nn.Module):
  def __init__(self):
    super().__init__()

    """ encoder """
    self.e1 = encoder(3, 128)
    self.e2 = encoder(128, 256)
    self.e3 = encoder(256, 512)
    self.e4 = encoder(512, 1024)

    """ Bridge Layer """
    self.b = conv_block(1024, 2048)

    """ decoder """
    self.d1 = decoder(2048, 1024)
    self.d2 = decoder(1024, 512)
    self.d3 = decoder(512, 256)
    self.d4 = decoder(256, 128)

    """ Output """
    self.outputs = nn.Conv2d(128, 1, kernel_size =1, padding=0)

  def forward(self, inputs):
    """ encoder """
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)

    """ Bridge """
    b = self.b(p4)

    """ decoder """
    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)

    """ Output """
    outputs = self.outputs(d4)
    # y = model_res34(inputs)
    # mask = torch.add(outputs, y)
    return outputs

model = Res34Unet()
model.load_state_dict(torch.load("C:/Users/lenovo/Desktop/VSCODE/django/retina_project/retina/machine/MODELS/Res34_Unet-3.pth", map_location=torch.device(device=device)))

# x = torch.randn((1,3,224,224))
# b = Res34Unet()
# y = model(x)
# print(y.shape)

def predict(img_path):
  im = cv2.imread(img_path, cv2.IMREAD_COLOR)
  im = cv2.resize(im, (224,224))
  im = np.transpose(im, (2,0,1))
  im = im/255.0
  im = np.expand_dims(im, axis=0)
  im = im.astype(np.float32)
  im = torch.from_numpy(im)
  im = im.to(device)
  print(im.shape)

  with torch.no_grad():
      pred_y = model(im)
      pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
      # print(pred_y.shape)
      pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
      # print(pred_y.shape)
      pred_y = pred_y > 0.5
      pred_y = np.array(pred_y, dtype=np.uint8)

  pred_y = mask_parse(pred_y)*255
  print(pred_y.shape)
  cv2.imwrite("C:/Users/lenovo/Desktop/VSCODE/django/retina_project/retina/static/results/pred.png", pred_y)
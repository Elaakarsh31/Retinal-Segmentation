import blocks
import torch
import torch.nn as nn
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

    """ blocks.encoder """
    self.e1 = blocks.encoder(3, 128)
    self.e2 = blocks.encoder(128, 256)
    self.e3 = blocks.encoder(256, 512)
    self.e4 = blocks.encoder(512, 1024)

    """ Bridge Layer """
    self.b = blocks.conv_block(1024, 2048)

    """ blocks.decoder """
    self.d1 = blocks.decoder(2048, 1024)
    self.d2 = blocks.decoder(1024, 512)
    self.d3 = blocks.decoder(512, 256)
    self.d4 = blocks.decoder(256, 128)

    """ Output """
    self.outputs = nn.Conv2d(128, 1, kernel_size =1, padding=0)

  def forward(self, inputs):
    """ blocks.encoder """
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)

    """ Bridge """
    b = self.b(p4)

    """ blocks.decoder """
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
model.load_state_dict(torch.load("MODELS/Res34_Unet-3.pth",map_location=torch.device(device=device)))

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
  cv2.imwrite("results/pred.png", pred_y)
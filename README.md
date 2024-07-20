# Overview
Image quality and processing power advances lead to the possibility of using these advancements in ophthalmology. Image processing and computer vision techniques are becoming highly significant in many sectors of science and medicine, and they are notably essential in modern ophthalmology because they primarily rely on visually oriented information. The optic microvasculature is a remarkable tissue as it is the only portion of the human vasculature that can be clearly depicted in vivo, photographed, and analyzed digitally. Over the last twenty years, a huge amount of research has been done on creating automated processes for retinal blood vessel segmentation. Despite this, delineation of retinal vessels remains a difficult process due to the existence of aberrations, non-uniform illumination, changing artery shape and size, and anatomical heterogeneity across patients. This research aims to give a thorough overview of retinal vessel segmentation techniques. I have used various techniques to improve the metrics results, such as Generative Adversarial Network (GAN), Image patches, Augmentation, and a combination of transfer learning and scratch U-Net models. The combination of transfer learning and U-Net models produced the best results in terms of IOU and dice score, accuracy, and F1-Score. The highest achieved accuracy score was 97 percent and the IoU Score was found to be 80 percent.

## Architecture of the model
![image](https://github.com/user-attachments/assets/be666901-5f4d-437c-ab74-28246a362c8d) ![image](https://github.com/user-attachments/assets/b6917c6e-4266-4283-8abf-99e8b0e9eb14)


# Retinal-Segmentation

Due to heavy size the model couldn't be uploaded. Train the cell under Res34/EfficientNet/Model-3 and save it as ModelRes34.h5 to use

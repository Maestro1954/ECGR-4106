# Libraries
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load Images
redImage = Image.open('/Horsehead.jpg')
greenImage = Image.open('/greenforest.jpg')
blueImage = Image.open('/Pleiades.jpg')

# Define transform for conversion
transform = transforms.ToTensor()

# Convert the image to tensor
tensor1 = transform(redImage)
tensor1 = tensor1.float()
tensor2 = transform(greenImage)
tensor2 = tensor2.float()
tensor3 = transform(blueImage)
tensor3 = tensor3.float()

# Mean brightness 1
print("***Red Image***")
brightness = torch.mean(tensor1)
print("Brightness: ", brightness)

# Mean of each channel RGB 1
R_mean, G_mean ,B_mean = torch.mean(tensor1, dim = [1,2])
print("Mean of Red channel:", R_mean)
print("Mean of Green channel:", G_mean)
print("Mean of Blue channel:", B_mean)

# Mean brightness 2
print("***Green Image***")
brightness = torch.mean(tensor2)
print("Brightness: ", brightness)

# Mean of each channel RGB 2
R_mean, G_mean ,B_mean = torch.mean(tensor2, dim = [1,2])
print("Mean of Red channel:", R_mean)
print("Mean of Green channel:", G_mean)
print("Mean of Blue channel:", B_mean)

# Mean brightness 3
print("***Blue Image***")
brightness = torch.mean(tensor3)
print("Brightness: ", brightness)

# Mean of each channel RGB 3
R_mean, G_mean ,B_mean = torch.mean(tensor3, dim = [1,2])
print("Mean of Red channel:", R_mean)
print("Mean of Green channel:", G_mean)
print("Mean of Blue channel:", B_mean)

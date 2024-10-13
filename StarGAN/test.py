import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model.deepfake.attribute_editing.StarGAN.stargan import Generator

#startGAN_generator = Generator(conv_dim=64, c_dim=5, repeat_num=6)
#startGAN_path = r"D:\python_project\2024\paper3\DefenseGAN\pretrained_models\StarGAN\model\200000-G.ckpt"
#startGAN_generator.load_state_dict(torch.load(startGAN_path, map_location=lambda storage, loc: storage))

#image_path = r'D:\python_project\2024\paper3\DefenseGAN\dataset\data\3.jpg'

# data_transforms = transforms.Compose(
#     [
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )

# label = torch.tensor([[0., 1., 0., 0., 0.]])
# image = Image.open(image_path).convert("RGB")
# image = data_transforms(image)


#out_ori = startGAN_generator(image.unsqueeze(0), label)
# out_ori = 0.5*255.*(out_ori[0]+1).detach().cpu().numpy().transpose(1, 2, 0)
# save_image(f'1.png', out_ori)
# print(out_ori)

def denormalize(tensor):
    return tensor * torch.tensor([0.5, 0.5, 0.5])[:, None, None] + torch.tensor([0.5, 0.5, 0.5])[:, None, None]

def save_image(output_tensor, path):
    output_tensor = denormalize(output_tensor.squeeze(0))
    output_image = transforms.ToPILImage()(output_tensor.cpu())
    output_image.save(path)

#save_image(out_ori, '2.png')
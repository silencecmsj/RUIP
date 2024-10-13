import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CelebHQ
import torch
import torchvision
from torchvision import transforms
import option
from StarGAN.stargan import Generator
from generator import UnetGenerator, get_norm_layer
from utils import Prepare_logger


def test(test_loader, defense_generator, global_noise, enhance):
    step = 0
    if not os.path.isdir('results/' + enhance):
        os.makedirs('./results/' + enhance + '/OI/')
        os.makedirs('./results/' + enhance + '/PI/')
        os.makedirs('./results/' + enhance + '/II/')
        os.makedirs('./results/' + enhance + '/FI/')

    for test_data in tqdm(test_loader):
        name, ori_image = test_data
        batch_size = ori_image.shape[0]
        ori_image = ori_image.to(device)

        # ---------------- generator ----------------------------------
        noise_image = torch.cat((ori_image, global_noise.unsqueeze(0).expand(batch_size, -1, -1, -1)), 1)
        noise = defense_generator(noise_image)
        protect_image = torch.clamp(ori_image + noise, -1, 1)
        enhance_ori_image = ori_image
        enhance_protected_image = protect_image

        for i in range(ori_image.shape[0]):
            step += 1
            fake_image = startgan_generator(enhance_ori_image,
                                            startgan_label[0].unsqueeze(0).expand(batch_size, -1).clone())
            invalid_enhance_image = startgan_generator(enhance_protected_image,
                                                       startgan_label[0].unsqueeze(0).expand(batch_size,
                                                                                             -1).clone())
            torchvision.utils.save_image((ori_image[i] + 1) / 2,
                                         'results/' + enhance + '/OI/' + str(step) + '.jpg')
            torchvision.utils.save_image((enhance_protected_image[i] + 1) / 2,
                                         'results/' + enhance + '/PI/' + str(step) + '.jpg')
            torchvision.utils.save_image((fake_image[i] + 1) / 2,
                                         'results/' + enhance + '/FI/' + str(step) + '.jpg')
            torchvision.utils.save_image((invalid_enhance_image[i] + 1) / 2,
                                         'results/' + enhance + '/II/' + str(step) + '.jpg')


if __name__ == '__main__':

    args = option.parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(args.checkpoint_path)
    norm_layer = get_norm_layer(norm_type='instance')
    defense_generator = UnetGenerator(6, 3, 6, ngf=64, norm_layer=norm_layer, use_dropout=True).to(device)
    # face manipulation model -- frozen
    logger = Prepare_logger(eval=False)
    logger.info('Loading model:{}'.format(args.checkpoint_path))

    startgan_generator = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    startgan_generator.load_state_dict(torch.load(args.stargan_path, map_location=lambda storage, loc: storage))
    for param in startgan_generator.parameters():
        param.requires_grad = False
    # Black_Hair Blond_Hair Brown_Hair Male Young
    startgan_label = torch.tensor([
                        [1., 0., 0., 0., 1.],
                        [0., 1., 0., 0., 1.],
                        [0., 0., 1., 0., 1.],
                        [0., 0., 0., 1., 1.],
                        [0., 0., 0., 0., 0.],
                     ]).to(device)

    defense_generator.load_state_dict(checkpoint['generator'])
    global_noise = checkpoint['noise'].to(device)
    enhance = 'none'

    data_transforms = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
    test_dataset = CelebHQ(args, 1, transforms=data_transforms)
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    logger.info('TestSet Number:{}'.format(test_dataset.num))

    test(test_loader, defense_generator, global_noise, enhance)
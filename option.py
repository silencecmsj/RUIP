import argparse

parser = argparse.ArgumentParser(description='Defense')

# config para
parser.add_argument('--csv_file_path', default=r'CelebA_HQ.csv', help='features csv file path')
parser.add_argument('--dataset', default="FF++", type=str, help='dataset Name')
parser.add_argument('--epoch', type=int, default=100, help='maximum iteration number (default: 100)')
parser.add_argument('--gpus', default="cuda:0", type=str, help='gpus')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--weight', type=float, default=0.2, help='weight_decay (default: 0.2)')
parser.add_argument('--checkpoint_path', type=str, default=r'checkpoint\stargan.tar', help='checkpoint path')
parser.add_argument('--batch_size', type=int, default=1, help='number of instances in a batch of data (default: 64)')
parser.add_argument('--status', type=int, default=0, help='whether to load the training model to continue training, 0:no 1:yes')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the crossnone-entropy objective used in the original GAN paper.')
# --- StarGAN ---
parser.add_argument('--stargan_path', type=str, default=r'checkpoint\200000-G.ckpt')

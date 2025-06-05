import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import StyledGenerator, Discriminator

from BreastCancerDS import *
import time

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
output_dir = f'run_wgan_run5'

try:
    os.mkdir(output_dir)
except: pass

# Turn on gradient
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# multiplies weight from one model (par2) into another model (par1)
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

# Create a dataloader for this epoch
def sample_data(dataset, batch_size=64, image_size=4):
    loader_kwargs = {'num_workers': 16, 'pin_memory': False, 'shuffle': True}
    dataset.NewResolution(image_size, batch_size)
    dataset.SetDropout(1.0)

    loader = DataLoader(dataset, batch_size=1, **loader_kwargs)
    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, epoch, dataset, generator, discriminator):
    # Tells us if we maxed out our upsample steps
    max_step = int(math.log2(args.max_size)) - 2
    step = epoch // 10
    if step > max_step:
        step = max_step
        final_progress = True
        ckpt_step = step + 1
    else:
        alpha = 0
        final_progress = False
        ckpt_step = step

    print ("===> Epoch, step, max step = ", epoch, step, max_step)

    step_batch_size = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 512, 256: 512}

    resolution = min(4 * 2 ** step , 512) # Step up the resolution until 512
    dataset.train()
    loader = sample_data( dataset, step_batch_size[resolution], resolution) # Get a new dataloader with updated parameters

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(len(loader)))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0



    i = 0
    for batch_idx, master_batch_data in enumerate(loader):

        master_features = master_batch_data[0].squeeze(0).cuda().float()

        for input_data in torch.split(master_features, 600):
            if input_data.shape[0] < 15: continue

            # MAIN STEP LOOP

            discriminator.zero_grad()
            i += 1

            # Oneline
            alpha = 1.0 if (resolution == args.init_size and args.ckpt is None) or final_progress else min(1, 1 / args.phase * (used_sample + 1))

            # Some book keeping
            real_image   = input_data
            used_sample += real_image.shape[0]
            b_size       = real_image.shape[0]
            real_image   = real_image.float()

            # Loss
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

            # Mixing
            if args.mixing and random.random() < 0.9:
                gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(4, b_size, code_size, device='cuda').chunk(4, 0)
                gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
                gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
            else:
                gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(2, 0)
                gen_in1 = gen_in1.squeeze(0)
                gen_in2 = gen_in2.squeeze(0)

            fake_image   = generator(gen_in1, step=step, alpha=alpha)
            fake_predict = discriminator(fake_image, step=step, alpha=alpha)

            # 'Grad Penalty
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()

            if (i + 1) % 10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

            d_optimizer.step()

            if (i + 1) % n_critic == 0:
                generator.zero_grad()

                requires_grad(generator, True)
                requires_grad(discriminator, False)

                fake_image = generator(gen_in2, step=step, alpha=alpha)
                predict = discriminator(fake_image, step=step, alpha=alpha)
                loss = (-predict).mean()


                if (i + 1) % 10 == 0:
                    gen_loss_val = loss.item()

                loss.backward()
                g_optimizer.step()
                accumulate(g_running, generator.module)

                requires_grad(generator, False)
                requires_grad(discriminator, True)

            if (i + 1) % 100 == 0:
                images = []

                gen_i, gen_j = args.gen_sample.get(resolution, (16, 8))

                with torch.no_grad():
                    for _ in range(gen_i):
                        images.append(
                            g_running(
                                torch.randn(gen_j, code_size).cuda(), step=step, alpha=alpha
                            ).data.cpu()
                        )
                utils.save_image(
                    torch.cat(images, 0),
                    f'{output_dir}/e{str(epoch)}_{str(i + 1).zfill(6)}_gen_running.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    fake_image[:gen_i*gen_j],
                    f'{output_dir}/e{str(epoch)}_{str(i + 1).zfill(6)}_gen_situ.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                )

            if (i + 1) % 100 == 0:
                gen_i, gen_j = args.gen_sample.get(resolution, (16, 8))

                utils.save_image(
                    real_image[:gen_i*gen_j],
                    f'{output_dir}/e{str(epoch)}_{str(i + 1).zfill(6)}_real.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                )

            state_msg = (
                f'Resolution: {resolution}; Seen Samples: {used_sample:d}; Gen Loss: {gen_loss_val:.8f}; Disc Loss: {disc_loss_val:.8f};'
                f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.5f}'
            )

            pbar.set_description(state_msg)
        pbar.update()
    pbar.close()
    # Done with epoch stuff
    torch.save(
        {
            'generator': generator.module.state_dict(),
            'discriminator': discriminator.module.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
            'g_running': g_running.state_dict(),
        },
        f'{output_dir}/train_step-{str(epoch)}.model',
    )

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))


if __name__ == '__main__':
    code_size = 256
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=512, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--epoch_start', default=0, type=int, help='Which epoch to start at?'
    )
    parser.add_argument(
        '--epoch_end', default=40, type=int, help='Which epoch to end at?'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    print (args)

    generator = nn.DataParallel(
        StyledGenerator(code_size),
        device_ids=[0, 1, 2, 3]
    ).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate),
        device_ids=[0, 1, 2, 3]
    ).cuda()

    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.2,
            'mult': 0.2,
        }
    )

    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        print ("Loading checkpoint!!")
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])


    dataset = BCSingleBagDatasetSimple(bag=True, output_dir=output_dir)
    dataset.load_new_train_only_flat()

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 512, 256: 512}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {256: (8, 4)}

    args.batch_default = 512

    for ep in range(args.epoch_start, args.epoch_end):
        print (ep)
        train(args, ep, dataset, generator, discriminator)

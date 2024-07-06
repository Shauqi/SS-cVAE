import torch
import torch.nn as nn
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None : 
            print("bias non None in : ", m)
            nn.init.constant_(m.bias.data, 0)
        else:
            print("bias None in : ", m)
    elif classname.find('Conv2d') != -1:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None : 
            print("bias non None in : ", m)
            nn.init.constant_(m.bias.data, 0)
        else:
            print("bias None in : ", m)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None : 
            print("bias non None in : ", m)
            nn.init.constant_(m.bias.data, 0)
        else:
            print("bias None in : ", m)

def create_model(args):
# Initialize generator and discriminator

    if args['discriminator']['model'] == 'DCGAN': 
        from .D_DCGAN import Discriminator
    else:
        raise NotImplementedError('Discriminator Model [%s] is not found' % args['discriminator']['model'])

    if args['generator']['model'] == 'DCGAN': 
        from .G_DCGAN import Generator
    else:
        raise NotImplementedError('Generator Model [%s] is not found' % args['generator']['model'])

    if args['cr']['model'] == 'DCGAN': 
        from .CR_DCGAN import CR_Net
    else:
        raise NotImplementedError('CR Model [%s] is not found' % args['cr']['model'])
        
   
    generator = Generator(args)
    discriminator = Discriminator(args)
    cr_net = CR_Net(args)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    cr_net.apply(weights_init_normal)

    # print("generator : ", generator)

    # print("discriminator : ", discriminator)

    # print("cr : ", cr_net)

    return generator, discriminator, cr_net

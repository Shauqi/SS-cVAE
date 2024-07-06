from gan_training.models import (
    resnet, resnet2, resnet3
)

# dvae_dict = {
#     'MM_cVAE': MM_cVAE.Conv_MM_cVAE,
#     'Guided_MM_cVAE': Guided_MM_cVAE.Conv_MM_cVAE,
# }

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet3': resnet3.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet3': resnet3.Discriminator,
}
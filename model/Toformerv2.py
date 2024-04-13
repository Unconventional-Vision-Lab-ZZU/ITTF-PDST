import torch.nn as nn
from model.Toformerv2_detail import Toformerv2


class Toformerv2_model(nn.Module):
    def __init__(self, config):
        super(Toformerv2_model, self).__init__()
        self.config = config
        self.Toformerv2 = Toformerv2(img_size=config['Toformerv2']['IMG_SIZE'],
                               patch_size=config['Toformerv2']['PATCH_SIZE'],
                               in_chans=config['Toformerv2']['in_chans'],
                               out_chans=config['Toformerv2']['out_chans'],
                               embed_dim=config['Toformerv2']['EMB_DIM'],
                               depths=config['Toformerv2']['DEPTH_EN'],
                               num_heads=config['Toformerv2']['HEAD_NUM'],
                               window_size=config['Toformerv2']['WIN_SIZE'],
                               mlp_ratio=config['Toformerv2']['MLP_RATIO'],
                               qkv_bias=config['Toformerv2']['QKV_BIAS'],
                               qk_scale=config['Toformerv2']['QK_SCALE'],
                               drop_rate=config['Toformerv2']['DROP_RATE'],
                               drop_path_rate=config['Toformerv2']['DROP_PATH_RATE'],
                               ape=config['Toformerv2']['APE'],
                               patch_norm=config['Toformerv2']['PATCH_NORM'],
                               use_checkpoint=config['Toformerv2']['USE_CHECKPOINTS'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.Toformerv2(x)
        return logits
    
if __name__ == '__main__':
    import torch
    import yaml
    from thop import profile

    ## Load yaml configuration file
    with open('../training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 256
    width = 256
    x = torch.randn((1, 156, height, width))  # .cuda()
    model = Toformerv2_model(opt)  # .cuda()
    out = model(x)
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)

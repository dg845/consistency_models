import argparse

import torch

from unet import UNetModel


def make_test_unet(num_classes=None):
    torch.manual_seed(0)
    unet = UNetModel(
        image_size=32,
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[2],
        dropout=0,
        channel_mult=(1, 2),
        conv_resample=True,
        dims=2,
        num_classes=num_classes,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=8,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )

    return unet


def main(args):
    unet = make_test_unet(num_classes=args.unet_num_classes)

    print(unet)
    unet_state_dict = unet.state_dict()
    for key in unet_state_dict:
        print(key)

    torch.save(unet.state_dict(), args.unet_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_output_path", default=None, type=str, required=True)
    parser.add_argument("--unet_num_classes", default=None, type=int)

    args = parser.parse_args()

    main(args)
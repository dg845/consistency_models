"""
Get test scheduler outputs with a simple dummy model.
"""

import argparse
import os
import json

import numpy as np
import torch as th

import logger
from script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from random_util import get_generator
from karras_diffusion import KarrasDenoiser, karras_sample


def create_diffusion(
    weight_schedule,
    sigma_min=0.002,
    sigma_max=80.0,
    distillation=False,
):
    diffusion = KarrasDenoiser(
        sigma_data=0.5,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        distillation=distillation,
        weight_schedule=weight_schedule,
    )
    return diffusion


def dummy_model():
    def model(sample, t, *args, **kwargs):
        # if t is a tensor, match the number of dimensions of sample
        if isinstance(t, th.Tensor):
            num_dims = len(sample.shape)
            # pad t with 1s to match num_dims
            t = t.reshape(-1, *(1,) * (num_dims - 1)).to(sample.device).to(sample.dtype)

        return sample * t / (t + 1)
    
    return model


def main():
    args = create_argparser().parse_args()
    d = vars(args)

    with open("config.json", "w") as fp:
        json.dump(d , fp)
    
    args = json.load(open("config.json"))

    logger.configure(dir="./logs")

    if "consistency" in args["training_mode"]:
        distillation = True
    else:
        distillation = False
    
    logger.log("creating model and diffusion...")
    diffusion = create_diffusion(
        args["weight_schedule"],
        sigma_max=args["sigma_max"],
        sigma_min=args["sigma_min"],
        distillation=distillation,
    )

    model = dummy_model()

    dev = 'cpu'

    logger.log("sampling...")
    if args["sampler"] == "multistep":
        assert len(args["ts"]) > 0
        ts = tuple(int(x) for x in args["ts"].split(","))
    else:
        ts = None
    
    all_images = []
    all_labels = []
    generator = get_generator(args["generator"], args["num_samples"], args["seed"])

    while len(all_images) * args["batch_size"] < args["num_samples"]:
        model_kwargs = {}
        if args["class_cond"]:
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args["batch_size"],), device=dev
            # )
            # Use generator to get random class labels for consistency
            # classes = generator.randint(
            #     low=0, high=NUM_CLASSES, size=(args["batch_size"],), device=dev
            # )
            # Use hard-coded class label
            classes = th.tensor([0], device=dev)
            # print(f"Classes: {classes}")
            # print(f"Classes shape: {classes.shape}")
            model_kwargs["y"] = classes
        
        sample = karras_sample(
            diffusion,
            model,
            (args["batch_size"], 3, args["image_size"], args["image_size"]),
            steps=args["steps"],
            model_kwargs=model_kwargs,
            device=dev,
            clip_denoised=args["clip_denoised"],
            sampler=args["sampler"],
            sigma_min=args["sigma_min"],
            sigma_max=args["sigma_max"],
            s_churn=args["s_churn"],
            s_tmin=args["s_tmin"],
            s_tmax=args["s_tmax"],
            s_noise=args["s_noise"],
            generator=generator,
            ts=ts,
        )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)

        # Get image slice
        print(f"Sample: {sample}")
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        sample_sum = th.sum(th.abs(sample))
        sample_mean = th.mean(th.abs(sample))
        print(f"Sample sum: {sample_sum}")
        print(f"Sample mean: {sample_mean}")

        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = sample.permute(0, 2, 3, 1)
        sample = (sample * 255).to(th.uint8)
        sample = sample.contiguous()

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images.extend([sample.cpu().numpy()])
        if args["class_cond"]:
            # gathered_labels = [
            #     th.zeros_like(classes) for _ in range(dist.get_world_size())
            # ]
            # dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            all_labels.extend([classes.cpu().numpy()])
        logger.log(f"created {len(all_images)} samples")
        # logger.log(f"created {len(all_images) * args["batch_size"]} samples"")
    
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args["num_samples"]]
    if args["class_cond"]:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args["num_samples"]]
    # if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args["class_cond"]:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="multistep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
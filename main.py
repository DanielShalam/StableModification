import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import utils
import ptp_utils
from null_inversion import NullInversion

MODEL = None
CONTEXT = None


def parse_args():
    """
    Command line argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="./configs/prompt2prompt.yaml"
    )
    parser.add_argument("--api_token", type=str, default="", help="hugging face token.")
    parser.add_argument("--outdir", type=str, default="./outputs/prompt2prompt/", )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--repeat", type=int, help="how many times to repeat the experiment with different seed.",
                        default=1, )
    opt = parser.parse_args()
    return opt


def get_prompts(exp_cfg):
    """
    Parsing config file
    """
    prompts = [exp_cfg["prompt_init"]] + list(exp_cfg["prompt_new"])
    return prompts


def get_source_img(exp_cfg, model, generator, prompt_src):
    """
    Generate/Invert source image
    """
    images_save = []
    # generating initial image
    if exp_cfg.img_path and exp_cfg.img_path != "":
        print("Perform Null text inversion")
        null_inversion = NullInversion(model, steps=exp_cfg.steps, guidance_scale=exp_cfg.scale)
        (image_gt, imgs_src), x_t, uncond_embeddings = null_inversion.invert(exp_cfg.img_path, prompt_src,
                                                                             offsets=(0, 0, 200, 0),
                                                                             verbose=True)
        images_save.append(ptp_utils.text_under_image(image_gt, text="source image"))
        images_save.append(ptp_utils.text_under_image(imgs_src.copy(), text="reconstructed image (inverted)"))
    else:
        print("Generate source image")
        uncond_embeddings = None
        controller = ptp_utils.AttentionStore()
        imgs_src, x_t = ptp_utils.text2image_ldm_stable(model, [prompt_src], controller, exp_cfg.steps,
                                                        latent=None, guidance_scale=exp_cfg.scale,
                                                        generator=generator)
        images_save.append(ptp_utils.text_under_image(imgs_src[0].copy(), text="source image"))

    return x_t, imgs_src, uncond_embeddings, images_save


def main():
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    base_count = len(os.listdir(args.outdir)) + 1
    model, tokenizer = utils.load_model_hugging(token=args.api_token, version=0)
    print(args)

    # load experiments from config file
    config = OmegaConf.load(f"{args.config}")
    experiments = config["experiments"]
    imgs_src = None
    uncond_embeddings = None
    x_t = None

    # iterate over experiments
    for exp_idx, exp_key in enumerate(experiments.keys()):
        exp_cfg = experiments[exp_key]
        prompts = get_prompts(exp_cfg)

        print(f"start experiment {exp_key} with prompts: ")
        print("prompt source:", prompts[0], "\nprompt target: ", prompts[1], "\n")

        for r in range(args.repeat):
            g_cpu = torch.Generator().manual_seed(args.seed + r)
            if exp_idx == 0 or not exp_cfg.resume_from_last:
                x_t, imgs_src, uncond_embeddings, images_save = get_source_img(exp_cfg, model, g_cpu, prompts[0])
            else:
                images_save = []

            # iterate over list of weights
            weights = [1] if not exp_cfg.reweight.apply else exp_cfg.reweight.weights
            for weight in weights:
                # get reweighing controller
                controller = utils.get_controller(exp_cfg, tokenizer, prompts, (float(weight),))

                # actual editing
                images, _ = ptp_utils.text2image_ldm_stable(model, prompts, controller, exp_cfg.steps, exp_cfg.scale,
                                                            latent=x_t.clone(), uncond_embeddings=uncond_embeddings)

                image = images[1] if np.ndim(images) > 3 else images
                images_text = ptp_utils.text_under_image(image, text=f"weight: {weight}")
                images_save.append(images_text)

            images_save_pil = ptp_utils.view_images(np.stack(images_save))
            images_save_pil.save(os.path.join(args.outdir, f"{base_count:05}.png"))
            base_count += 1


if __name__ == '__main__':
    args = parse_args()
    main()

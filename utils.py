import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler,\
    StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler

import ptp_utils
from ldm.util import instantiate_from_config

model_ids = ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_v2_modules():
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(model_ids[1], subfolder="vae").to(device)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(model_ids[1], subfolder="unet").to(device)
    return vae, tokenizer, text_encoder, unet


def load_model_hugging(token: str = None, version: int = 0):
    """
    Load pretrained diffusion model from hugging-face
    """
    assert version < 2
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    model = StableDiffusionPipeline.from_pretrained(model_ids[version], use_auth_token=token, scheduler=scheduler,
                                                    torch_dtype=torch.float32).to(device)
    tokenizer = model.tokenizer
    return model, tokenizer


def get_controller(cfg, tokenizer, prompts, max_num_words=77, weight=None):
    """
    Get Prompt2Prompt controller
    """
    assert cfg.replace.apply is not cfg.refine.apply, print("Only one from replace/refine should be selected.")

    local_blend = None
    controller = None
    if cfg.localize.apply:
        local_blend = ptp_utils.LocalBlend(tokenizer, prompts, words=tuple(cfg.localize.words),
                                           max_num_words=max_num_words)

    if cfg.replace.apply:
        controller = ptp_utils.AttentionReplace(
            tokenizer, prompts, cfg.steps, cfg.cross_replace_steps,
            self_replace_steps=cfg.self_replace_steps, local_blend=local_blend
        )
    elif cfg.refine.apply:
        controller = ptp_utils.AttentionRefine(
            tokenizer, prompts, cfg.steps, cfg.cross_replace_steps,
            self_replace_steps=cfg.self_replace_steps, local_blend=local_blend
        )

    if cfg.reweight.apply:
        to_reweight = cfg.reweight.text
        controller = reweight_controller(tokenizer, prompts, controller, local_blend, to_reweight, weight, cfg.steps)

    return controller


def replace_controller(tokenizer, prompts, lb=None, steps=50):
    """
    Generate controller for image replacement
    """

    controller = ptp_utils.AttentionReplace(
        tokenizer, prompts, steps, cross_replace_steps={"default_": 1., "green": .4},
        self_replace_steps=0.6, local_blend=lb
    )

    # controller = ptp_utils.AttentionReplace(
    #     tokenizer, prompts, steps, cross_replace_steps={"default_": 1., "lasagne": .2},
    #     cross_replace_steps=.8, self_replace_steps=0.4, local_blend=lb)
    return controller


def reweight_controller(tokenizer, prompts, controller, lb, to_reweight, weights, steps=50):
    """
    Generate controller for image reweight/refinement
    """

    # if weights = 10 and to_reweight="fried", we pay 10 times more attention to the word "fried"
    equalizer = ptp_utils.get_equalizer(tokenizer, prompts[1], to_reweight, weights)
    controller = ptp_utils.AttentionReweight(tokenizer, prompts, steps, cross_replace_steps=.8, self_replace_steps=.4,
                                             equalizer=equalizer, local_blend=lb, controller=controller)
    return controller


def load_model_from_config(config_path="./configs/stable-diffusion/v2-inference-v.yaml",
                           ckpt_path="./models/v2-1_768-ema-pruned.ckpt", verbose=False,
                           return_tokenizer=False):
    """
    Load local pretrained model from config
    """
    config = OmegaConf.load(f"{config_path}")

    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    print(f"Model type: {type(model)}")
    model.cuda()
    model.eval()
    model = model.to(device)
    print(model)
    if return_tokenizer:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        return model, tokenizer

    return model

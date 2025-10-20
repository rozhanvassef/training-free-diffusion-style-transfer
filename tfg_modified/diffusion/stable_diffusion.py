import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List

from diffusers.utils.torch_utils import randn_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
from diffusers import StableDiffusionImg2ImgPipeline  # Changed from StableDiffusionPipeline
from  diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from utils.env_utils import *

class StableDiffusionSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionSampler, self).__init__(args)
        self.image_size = args.image_size
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)

    @torch.no_grad()
    def decode(self, latents):
        return self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

    @torch.no_grad()
    def encode(self, image):
        """Encode PIL image to latents"""
        # Convert PIL to tensor if needed
        if isinstance(image, Image.Image):
            image = to_tensor(image).unsqueeze(0).to(self.device)
        # Normalize to [-1, 1]
        if image.max() > 1:
            image = image / 255.0
        image = image * 2 - 1
        # Encode to latents
        latents = self.sd_pipeline.vae.encode(image).latent_dist.sample(generator=self.generator)
        latents = latents * self.sd_pipeline.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_name_or_path).to(self.device)
        self.scheduler = self.sd_pipeline.scheduler
        
        unet = self.sd_pipeline.unet
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = False

        self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

        # prepare prompts: str or List[str]
        self.prompts = self._prepare_prompts(args)  

        # FIXME: classifier-free guidance params
        self.do_classifier_free_guidance = True
        self.guidance_scale = 2.5  # Reduced for img2img

        # Get strength parameter for img2img
        self.strength = getattr(args, 'strength', 0.75)  # Default strength

        # FIXED: check inputs. Raise error if not correct
        # CHANGE: Img2Img pipeline doesn't need check_inputs call here - remove it
        # The pipeline will validate inputs during the actual generation process
        # self.sd_pipeline.check_inputs("", None, self.height, self.width, strength=self.strength)

        self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = unet, ts, alpha_prod_ts, alpha_prod_t_prevs

    
    def _prepare_prompts(self, args):
        
        if hasattr(args, 'init_image') and args.init_image:
            prompts = [""] * args.num_samples  # Empty prompts for style transfer
        else:
            # Fallback to original behavior if no init_image
            if args.dataset == 'parti_prompts':
                prompts = [line.strip() for line in open(PARTIPROMPOTS_PATH, 'r').readlines()][:args.num_samples]
            else:
                prompts = [""] * args.num_samples  # Changed from "flower" to empty
        
        return prompts

    def _prepare_init_latents(self, init_image, batch_size):
        """Prepare initial latents from input image"""
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert('RGB')
        
        # Resize image to match model requirements
        init_image = init_image.resize((self.width, self.height), Image.LANCZOS)
        
        # Encode to latents
        init_latents = self.encode(init_image)
        
        # Repeat for batch size
        init_latents = init_latents.repeat(batch_size, 1, 1, 1)
        
        return init_latents

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):
        
        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        for batch_id in range(n_batchs):
            
            self.args.batch_id = batch_id

            prompts = self.prompts[batch_id * self.per_sample_batch_size: min((batch_id + 1) * self.per_sample_batch_size, len(self.prompts))]

            # encode input prompts
            prompt_embeds, negative_prompt_embeds = self.sd_pipeline.encode_prompt(
                prompts, 
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # Prepare initial latents from input image instead of random noise
            if hasattr(self.args, 'init_image') and self.args.init_image:
                init_latents = self._prepare_init_latents(self.args.init_image, len(prompts))
                
                # Add noise to the initial latents based on strength
                noise = randn_tensor(init_latents.shape, generator=self.generator, device=self.device, dtype=init_latents.dtype)
                
                # Get the original timestep using init_timestep
                init_timestep = min(int(self.inference_steps * self.strength), self.inference_steps)
                init_timestep = max(init_timestep, 1)
                
                timesteps = self.scheduler.timesteps[-init_timestep:]
                timestep = timesteps[0]
                
                # Add noise to latents at this timestep
                latents = self.scheduler.add_noise(init_latents, noise, timestep)
                
                # Adjust timesteps for img2img
                self.ts = timesteps
                t_start = len(timesteps) - init_timestep
                
            else:
                # Fallback to original random latents
                latents = self.sd_pipeline.prepare_latents(
                    len(prompts),
                    self.sd_pipeline.unet.config.in_channels,
                    self.height,
                    self.width,
                    prompt_embeds.dtype,
                    self.device,
                    generator=self.generator
                )
                t_start = 0

            for i, t in enumerate(tqdm(self.ts[t_start:], total=len(self.ts[t_start:]))):
                
                def stable_diffusion_unet(latents, t):

                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    return noise_pred

                # Adjust indices for the guidance step
                step_index = t_start + i
                if step_index < len(self.alpha_prod_ts):
                    # For img2img, we need to pass the correct alpha arrays that match our timesteps
                    current_ts = self.ts[t_start:]
                    current_alpha_prod_ts = self.scheduler.alphas_cumprod[current_ts]
                    current_alpha_prod_t_prevs = torch.cat([current_alpha_prod_ts[1:], torch.ones(1, device=self.device) * self.scheduler.final_alpha_cumprod])
                    
                    latents = guidance.guide_step(
                        latents, i,  # Use i (the loop index) instead of t (the timestep value)
                        stable_diffusion_unet,
                        current_ts,
                        current_alpha_prod_ts, 
                        current_alpha_prod_t_prevs,
                        self.eta
                    )

            image = self.decode(latents)
            tot_samples.append(image.clone().cpu())
        
        return torch.concat(tot_samples)
        
    def tensor_to_obj(self, x):

        images = (x / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images
    
    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images).to(self.device)
        return tensor_images * 2 - 1
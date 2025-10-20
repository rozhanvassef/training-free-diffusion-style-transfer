#THIS FILE HAS ALSO BEEN CHANGED AFTER DELETING SOME OTHER FILES.
#BUT THE PREVIOUS VERSION IS NOT HERE.

import functools
import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List, Any, Union, Tuple
import numpy as np

from diffusers.utils.torch_utils import randn_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
import logger

# REMOVED: All molecule-related imports since we deleted molecule support
# from tasks.networks.qm9 import dataset  # DELETED
# from tasks.networks.qm9.utils import compute_mean_mad  # DELETED
# from tasks.networks.qm9.datasets_config import get_dataset_info  # DELETED
# from tasks.networks.qm9.models import DistributionProperty, DistributionNodes  # DELETED
# from tasks.networks.egnn.EDM import EDM  # DELETED
# from tasks.networks.egnn.EGNN import EGNN_dynamics_QM9  # DELETED
# from tasks.networks.egnn.utils import assert_correctly_masked, assert_mean_zero_with_mask  # DELETED


class ImageSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(ImageSampler, self).__init__(args)
        self.object_size = (3, args.image_size, args.image_size)
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)
        self.target = args.target

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)

    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        if 'openai' in args.model_name_or_path:
            from .unet.openai import get_diffusion
        else: 
            from .unet.huggingface import get_diffusion
        
        self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = get_diffusion(args)
    

    @torch.no_grad()
    def sample(self, sample_size: int, guidance: BaseGuidance):
        
        tot_samples = []
        n_batchs = math.ceil(sample_size / self.per_sample_batch_size)

        for batch_id in range(n_batchs):
            
            self.args.batch_id = batch_id

            x = randn_tensor(
                shape=(self.per_sample_batch_size, *self.object_size),
                generator=self.generator,
                device=self.device,
            )

            for t in tqdm(range(self.inference_steps), total=self.inference_steps):
                
                x = guidance.guide_step(
                    x, t, self.unet,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta,
                )

                # we may want to log some trajs
                if self.log_traj:
                    logger.log_samples(self.tensor_to_obj(x), fname=f'traj/time={t}')

            tot_samples.append(x)
        
        return torch.concat(tot_samples)
        
    @staticmethod
    def tensor_to_obj(x):

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

    @staticmethod
    def obj_to_tensor(objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images)
        return tensor_images * 2 - 1


# REMOVED: Entire MoleculeSampler class since we deleted molecule support
# The MoleculeSampler class and all its methods have been completely removed
# because it depends on the deleted egnn and qm9 modules
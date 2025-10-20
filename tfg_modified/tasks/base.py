'''
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk, load_dataset
from diffusers import StableDiffusionPipeline
from functools import partial
import logger

from .image_label_guidance import ImageLabelGuidance
from .style_transfer_guidance import StyleTransferGuidance
from .super_resolution import SuperResolution
from .gaussian_deblur import GaussianDeblur
from .molecule_properties import MoleculePropertyGuidance
from .audio_declipping import AduioDeclippingGuidance
from .audio_inpainting import AduioInpaintingGuidance

class BaseGuider:

    def __init__(self, args):
        self.args = args
        self.generator = torch.manual_seed(args.seed)
        
        self.load_processor()   # e.g., vae for latent diffusion
        self.load_guider()      # guidance network

    def load_processor(self):
        if self.args.data_type == 'text2image':
            sd = StableDiffusionPipeline.from_pretrained(self.args.model_name_or_path)
            self.vae = sd.vae
            self.vae.eval()
            self.vae.to(self.args.device)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.processor = lambda x: self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
        else:
            self.processor = lambda x: x

    @torch.enable_grad()
    def process(self, x):
        return self.processor(x)

    @torch.no_grad()
    def load_guider(self):
        
        self.get_guidance = None

        # for combined guidance
        device = self.args.device

        guiders = []

        for task, guide_network, target in zip(self.args.tasks, self.args.guide_networks, self.args.targets):

            if task == 'style_transfer':
                guider = StyleTransferGuidance(guide_network, target, device)
            elif task == 'label_guidance':
                guider = ImageLabelGuidance(guide_network, target, device, time=False)
            elif task == 'label_guidance_time':
                guider = ImageLabelGuidance(guide_network, target, device, time=True)
            elif task == 'super_resolution':
                guider = SuperResolution(self.args)
            elif task == 'gaussian_deblur':
                guider = GaussianDeblur(self.args)
            elif task == 'molecule_property':
                guider = MoleculePropertyGuidance(self.args)
            elif task == 'audio_declipping':
                guider = AduioDeclippingGuidance(self.args)
            elif task == 'audio_inpainting':
                guider = AduioInpaintingGuidance(self.args)
            else:
                raise NotImplementedError
            
            guiders.append(guider)
        
        if len(guiders) == 1:
            self.get_guidance = partial(guider.get_guidance, post_process=self.process)
        else:
            self.get_guidance = partial(self._get_combined_guidance, guiders=guiders)

        if self.get_guidance is None:
            raise ValueError(f"Unknown guider: {self.args.guider}")
    
    def _get_combined_guidance(self, x, guiders, *args, **kwargs):
        values = []
        for guider in guiders:
            values.append(guider.get_guidance(x, post_process=self.process, *args, **kwargs))
        return sum(values)
'''
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from datasets import load_from_disk, load_dataset
from diffusers import StableDiffusionPipeline
from functools import partial
import logger

# REMOVED: All imports of deleted task files
# from .image_label_guidance import ImageLabelGuidance  # DELETED
# from .super_resolution import SuperResolution  # DELETED
# from .gaussian_deblur import GaussianDeblur  # DELETED
# from .molecule_properties import MoleculePropertyGuidance  # DELETED
# from .audio_declipping import AduioDeclippingGuidance  # DELETED
# from .audio_inpainting import AduioInpaintingGuidance  # DELETED

# KEPT: Only import existing style transfer guidance
from .style_transfer_guidance import StyleTransferGuidance

class BaseGuider:
    def __init__(self, args):
        self.args = args
        self.generator = torch.manual_seed(args.seed)
        self.load_processor()  # e.g., vae for latent diffusion
        self.load_guider()  # guidance network

    def load_processor(self):
        if self.args.data_type == 'text2image':
            sd = StableDiffusionPipeline.from_pretrained(self.args.model_name_or_path)
            self.vae = sd.vae
            self.vae.eval()
            self.vae.to(self.args.device)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.processor = lambda x: self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]
        else:
            self.processor = lambda x: x

    @torch.enable_grad()
    def process(self, x):
        return self.processor(x)

    @torch.no_grad()
    def load_guider(self):
        self.get_guidance = None
        # for combined guidance
        device = self.args.device
        guiders = []
        
        for task, guide_network, target in zip(self.args.tasks, self.args.guide_networks, self.args.targets):
            # KEPT: Only style transfer guidance
            if task == 'style_transfer':
                guider = StyleTransferGuidance(guide_network, target, device)
            # NEW: Added support for point cloud guidance
            elif task == 'pointcloud_guidance':
                # Point cloud guidance will be handled by TFG wrapper in utils/utils.py
                # This is just a placeholder to avoid errors
                print("Point cloud guidance detected - will be handled by TFG wrapper")
                continue
            # REMOVED: All other task types that were deleted
            # elif task == 'label_guidance':
            #     guider = ImageLabelGuidance(guide_network, target, device, time=False)  # DELETED
            # elif task == 'label_guidance_time':
            #     guider = ImageLabelGuidance(guide_network, target, device, time=True)  # DELETED
            # elif task == 'super_resolution':
            #     guider = SuperResolution(self.args)  # DELETED
            # elif task == 'gaussian_deblur':
            #     guider = GaussianDeblur(self.args)  # DELETED
            # elif task == 'molecule_property':
            #     guider = MoleculePropertyGuidance(self.args)  # DELETED
            # elif task == 'audio_declipping':
            #     guider = AduioDeclippingGuidance(self.args)  # DELETED
            # elif task == 'audio_inpainting':
            #     guider = AduioInpaintingGuidance(self.args)  # DELETED
            else:
                # MODIFIED: Better error message for unsupported tasks
                supported_tasks = ['style_transfer', 'pointcloud_guidance']
                raise NotImplementedError(f"Task '{task}' not supported. Available tasks: {supported_tasks}")
            
            guiders.append(guider)

        # MODIFIED: Handle case when no guiders are added (e.g., for point cloud guidance)
        if len(guiders) == 0:
            print("No traditional guiders loaded - using TFG wrapper approach")
            return
        elif len(guiders) == 1:
            self.get_guidance = partial(guiders[0].get_guidance, post_process=self.process)
        else:
            self.get_guidance = partial(self._get_combined_guidance, guiders=guiders)

        # REMOVED: This error check since we now allow no guiders for point cloud case
        # if self.get_guidance is None:
        #     raise ValueError(f"Unknown guider: {self.args.guider}")

    def _get_combined_guidance(self, x, guiders, *args, **kwargs):
        """Combine guidance from multiple guiders"""
        values = []
        for guider in guiders:
            values.append(guider.get_guidance(x, post_process=self.process, *args, **kwargs))
        return sum(values)
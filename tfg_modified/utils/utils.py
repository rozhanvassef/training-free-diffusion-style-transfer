'''
import torch
import os
from typing import Union
from transformers import HfArgumentParser

from .configs import Arguments

from evaluations.image import ImageEvaluator
from evaluations.molecule import MoleculeEvaluator
from evaluations.audio import AudioEvaluator

from diffusion.ddim import ImageSampler, MoleculeSampler
from diffusion.audio_diffusion import AudioDiffusionSampler
from diffusion.stable_diffusion import StableDiffusionSampler

from methods.mpgd import MPGDGuidance
from methods.lgd import LGDGuidance
from methods.base import BaseGuidance
from methods.ugd import UGDGuidance
from methods.freedom import FreedomGuidance
from methods.dps import DPSGuidance
from methods.tfg import TFGGuidance
from methods.cg import ClassifierGuidance


import pickle


def get_logging_dir(arg_dict: dict):
    if arg_dict['guidance_name'] == 'tfg':
        # record rho, mu, sigma with scheduler
        suffix = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
    else:
        suffix = "guidance_strength=" + str(arg_dict['guidance_strength'])
    
    return os.path.join(
        arg_dict['logging_dir'],
        f"guidance_name={arg_dict['guidance_name']}+recur_steps={arg_dict['recur_steps']}+iter_steps={arg_dict['iter_steps']}",
        "model=" + arg_dict['model_name_or_path'].replace("/", '_'),
        "guide_net=" + arg_dict['guide_network'].replace('/', '_'),
        "target=" + str(arg_dict['target']).replace(" ", "_"),
        "bon=" + str(arg_dict['bon_rate']),
        suffix,
    )

def get_config(add_logger=True) -> Arguments:
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)

    if add_logger:
        from logger import setup_logger
        args.logging_dir = get_logging_dir(vars(args))
        print("logging to", args.logging_dir)
        setup_logger(args)
    
    if args.data_type == 'molecule':
        # load args
        def _get_args_gen(args_path, argse_path):
            with open(args_path, 'rb') as f:
                args_gen = pickle.load(f)
            assert args_gen.dataset == 'qm9_second_half'

            with open(argse_path, 'rb') as f:
                args_en = pickle.load(f)

            # Add missing args!
            if not hasattr(args_gen, 'normalization_factor'):
                args_gen.normalization_factor = 1
            if not hasattr(args_gen, 'aggregation_method'):
                args_gen.aggregation_method = 'sum'

            return args_gen, args_en

        args.args_gen, args.args_en = _get_args_gen(args.args_generators_path, args.args_energy_path)
    
    # examine combined guidance

    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')

    assert len(args.tasks) == len(args.guide_networks) == len(args.targets)

    return args


def get_evaluator(args):

    if args.data_type == 'image':
        return ImageEvaluator(args)
    elif args.data_type == 'molecule':
        return MoleculeEvaluator(args)
    elif args.data_type == 'text2image':
        return ImageEvaluator(args)
    elif args.data_type == 'audio':
        return AudioEvaluator(args)
    else:
        raise NotImplementedError

def get_guidance(args, network):
    noise_fn = getattr(network, 'noise_fn', None)
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'mpgd':
        return MPGDGuidance(args, noise_fn=noise_fn)
    elif 'ugd' in args.guidance_name:
        return UGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'freedom':
        return FreedomGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'dps':
        return DPSGuidance(args, noise_fn=noise_fn)
    elif 'lgd' in args.guidance_name:
        return LGDGuidance(args, noise_fn=noise_fn)
    elif "tfg" in args.guidance_name:
        return TFGGuidance(args, noise_fn=noise_fn)
    elif 'cg' in args.guidance_name:
        return ClassifierGuidance(args, noise_fn=noise_fn)
    else:
        raise NotImplementedError

def get_network(args):
    
    if args.data_type == 'image':
        return ImageSampler(args)
    elif args.data_type == 'molecule':
        return MoleculeSampler(args)
    elif args.data_type == 'text2image':
        return StableDiffusionSampler(args)
    elif args.data_type == 'audio':
        return AudioDiffusionSampler(args)
    else:
        raise NotImplementedError
'''

import torch
import os
from typing import Union
from transformers import HfArgumentParser

from .configs import Arguments

# FIXED: Only import existing evaluators
from evaluations.image import ImageEvaluator
# REMOVED: from evaluations.molecule import MoleculeEvaluator  # DELETED
# REMOVED: from evaluations.audio import AudioEvaluator  # DELETED

# FIXED: Only import existing diffusion samplers
from diffusion.ddim import ImageSampler
from diffusion.stable_diffusion import StableDiffusionSampler
# REMOVED: from diffusion.ddim import MoleculeSampler  # DELETED
# REMOVED: from diffusion.audio_diffusion import AudioDiffusionSampler  # DELETED

# FIXED: Only import existing guidance methods
from methods.base import BaseGuidance
from methods.tfg import TFGGuidance
# REMOVED: All other guidance methods that were deleted

import pickle


def get_logging_dir(arg_dict: dict):
    if arg_dict['guidance_name'] == 'tfg':
        # record rho, mu, sigma with scheduler
        suffix = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
    else:
        suffix = "guidance_strength=" + str(arg_dict['guidance_strength'])
    
    return os.path.join(
        arg_dict['logging_dir'],
        f"guidance_name={arg_dict['guidance_name']}+recur_steps={arg_dict['recur_steps']}+iter_steps={arg_dict['iter_steps']}",
        "model=" + arg_dict['model_name_or_path'].replace("/", '_'),
        "guide_net=" + arg_dict['guide_network'].replace('/', '_'),
        "target=" + str(arg_dict['target']).replace(" ", "_"),
        "bon=" + str(arg_dict['bon_rate']),
        suffix,
    )

def get_config(add_logger=True) -> Arguments:
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)

    if add_logger:
        from logger import setup_logger
        args.logging_dir = get_logging_dir(vars(args))
        print("logging to", args.logging_dir)
        setup_logger(args)
    
    # REMOVED: Molecule-specific code since we deleted molecule support
    
    # examine combined guidance
    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')

    assert len(args.tasks) == len(args.guide_networks) == len(args.targets)

    return args


def get_evaluator(args):
    # FIXED: Only support remaining data types
    if args.data_type == 'image':
        return ImageEvaluator(args)
    elif args.data_type == 'text2image':
        return ImageEvaluator(args)
    else:
        raise NotImplementedError(f"Data type '{args.data_type}' not supported. Available: 'image', 'text2image'")

def get_guidance(args, network):
    noise_fn = getattr(network, 'noise_fn', None)
    
    # FIXED: Only support remaining guidance methods
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif "tfg" in args.guidance_name:
        return TFGGuidance(args, noise_fn=noise_fn)
    # NEW: Add support for style transfer guidance
    elif args.guidance_name == 'style_transfer':
        from tasks.style_transfer_guidance import StyleTransferGuidance
        return TFGGuidance(args, noise_fn=noise_fn, 
                          guider=StyleTransferGuidance(
                              guide_network=args.guide_networks[0],
                              target=args.targets[0],
                              device=args.device
                          ))
    # NEW: Add support for point cloud guidance
    elif args.guidance_name == 'pointcloud_l1':
        from tasks.pointcloud_guidance import PointCloudL1Guidance
        
        # Load camera pose if provided
        camera_pose = None
        if hasattr(args, 'pcd_camera_pose') and args.pcd_camera_pose is not None:
            print(f"Loading camera pose from: {args.pcd_camera_pose}")
            camera_pose = torch.load(args.pcd_camera_pose) if args.pcd_camera_pose.endswith('.pt') else np.load(args.pcd_camera_pose)
            
        # Validate point cloud file path
        if not hasattr(args, 'pcd_file_path') or args.pcd_file_path is None:
            raise ValueError("ERROR: --pcd_file_path must be specified for point cloud guidance!")
            
        if not os.path.exists(args.pcd_file_path):
            raise FileNotFoundError(f"ERROR: Point cloud file not found: {args.pcd_file_path}")
            
        print(f"Initializing Point Cloud L1 Guidance:")
        print(f"  - PCD file: {args.pcd_file_path}")
        print(f"  - Image size: {getattr(args, 'pcd_image_size', [512, 512])}")
        print(f"  - Focal length: {getattr(args, 'pcd_focal_length', 500.0)}")
        print(f"  - Camera pose: {'Custom' if camera_pose is not None else 'Default'}")
        
        pointcloud_guidance = PointCloudL1Guidance(
            pcd_file_path=args.pcd_file_path,
            camera_pose=camera_pose,
            image_size=tuple(getattr(args, 'pcd_image_size', [512, 512])),
            focal_length=getattr(args, 'pcd_focal_length', 500.0),
            device=args.device
        )
        
        return TFGGuidance(args, noise_fn=noise_fn, guider=pointcloud_guidance)
    else:
        raise NotImplementedError(f"Guidance method '{args.guidance_name}' not supported. Available: 'no', 'tfg', 'style_transfer', 'pointcloud_l1'")

def get_network(args):
    # FIXED: Only support remaining network types
    if args.data_type == 'image':
        return ImageSampler(args)
    elif args.data_type == 'text2image':
        return StableDiffusionSampler(args)
    else:
        raise NotImplementedError(f"Network type '{args.data_type}' not supported. Available: 'image', 'text2image'")
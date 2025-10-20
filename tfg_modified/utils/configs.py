'''
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List
from logger.base import BaseLogger

@dataclass
class Arguments:
    # data related
    data_type: Literal['image', 'molecule', 'text2image', 'audio', 'motion'] = field(default='image')
    dataset: str = field(default='cifar10')
    task: str = field(default='label_guidance')
    
    # image related
    image_size: int = field(default=32)
    
    # CHANGED: Uncommented these lines for IMG2IMG functionality
    init_image: Optional[str] = field(default=None)  # Path to input image for img2img
    strength: float = field(default=0.75)  # Noise strength (0-1), higher = more creative freedom
    
    # molecule related
    include_charges: bool = field(default=False)
    generators_path: str = field(default='./pretrained_models/EDMsecond/generative_model_ema.npy')
    args_generators_path: str = field(default='./pretrained_models/EDMsecond/args.pickle')
    energy_path: str = field(default='./pretrained_models/tf_predict_mu/model_ema_2000.npy')
    args_energy_path: str = field(default='./pretrained_models/tf_predict_mu/args_2000.pickle')
    classifiers_path: str = field(default='./pretrained_models/evaluate_mu/best_checkpoint.npy')
    args_classifiers_path: str = field(default='./pretrained_models/evaluate_mu/args.pickle')
    clip_scale: float = field(default=100)
    
    # audio related
    audio_length: int = field(default=10)
    volume_factor: float = field(default=80.0)
    
    # motion related
    motion_prompt: str = field(default='walk')
    radius: float = field(default=3.0)
    
    # model related
    model_name_or_path: str = field(default='google/ddpm-cifar10-32')
    
    # diffusion related
    train_steps: int = field(default=1000)
    inference_steps: int = field(default=50)
    eta: float = field(default=1.0)
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)
    
    # inference related:
    seed: int = field(default=32)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    logger: BaseLogger = None # Initialize upon instantiation
    per_sample_batch_size: int = field(default=128)
    num_samples: int = field(default=2048)
    batch_id: int = field(default=0) # start from the zero
    
    # guidance related
    guidance_name: str = field(default='no')
    guider: str = field(default='classifier')
    target: str = field(default=None)
    recur_steps: int = field(default=1)
    iter_steps: int = field(default=1)
    guidance_strength: float = field(default=1.0)
    
    # specific for our method
    rho: float = field(default=1.0)
    mu: float = field(default=1.0)
    sigma: float = field(default=0.01)
    eps_bsz: int = field(default=4)
    rho_schedule: str = field(default='increase')
    mu_schedule: str = field(default='increase')
    sigma_schedule: str = field(default='decrease')
    
    # cond_fn related
    guide_network: str = field(default='aaraki/vit-base-patch16-224-in21k-finetuned-cifar10')
    classifier_image_size: int = field(default=224)
    
    # evaluation related
    eval_batch_size: int = field(default=128)
    
    # logging related
    logging_resolution: int = field(default=64)
    log_suffix: str = field(default='')
    log_traj: bool = field(default=False)
    max_show_images: int = field(default=256)
    check_done: bool = field(default=True)
    
    # wandb related
    wandb: bool = field(default=False)
    wandb_project: str = field(default='trail')
    wandb_name: str = field(default=None)
    wandb_entity: str = field(default='llm-selection')
    
    # visualization related
    saved_file: str = field(default=None)
    sort_metric: str = field(default=None)
    topk: int = field(default=5)
    output_path: str = field(default='vis_molecule')

    max_n_samples: int = field(default=10000000000)
    """
<<<<<<< HEAD
    max_n_samples: int = field(default=10000000000)
    
    # bon
    bon_rate: float = field(default=1)
    bon_guidance: str = field(default=None)

=======
    max_n_samples: int = field(default=10000000000)
>>>>>>> my-changes
    """
'''
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, List
from logger.base import BaseLogger

@dataclass
class Arguments:
    # data related
    data_type: Literal['image', 'text2image'] = field(default='image')  # FIXED: Removed unsupported types
    dataset: str = field(default='cifar10')
    task: str = field(default='label_guidance')
    
    # image related
    image_size: int = field(default=32)
    
    # CHANGED: Uncommented these lines for IMG2IMG functionality
    init_image: Optional[str] = field(default=None)  # Path to input image for img2img
    strength: float = field(default=0.75)  # Noise strength (0-1), higher = more creative freedom
    
    # REMOVED: molecule related fields (deleted molecule support)
    # REMOVED: audio related fields (deleted audio support)
    # REMOVED: motion related fields (deleted motion support)
    
    # NEW: Point cloud guidance arguments
    pcd_file_path: Optional[str] = field(default=None, metadata={'help': 'Path to the .pcd point cloud file for guidance'})
    pcd_camera_pose: Optional[str] = field(default=None, metadata={'help': 'Path to camera pose file (.npy) or None for default view'})
    pcd_image_size: List[int] = field(default_factory=lambda: [512, 512], metadata={'help': 'Target image size for point cloud rendering [width, height]'})
    pcd_focal_length: float = field(default=500.0, metadata={'help': 'Focal length for point cloud camera projection'})
    
    # model related
    model_name_or_path: str = field(default='google/ddpm-cifar10-32')
    
    # diffusion related
    train_steps: int = field(default=1000)
    inference_steps: int = field(default=50)
    eta: float = field(default=1.0)
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)
    
    # inference related:
    seed: int = field(default=32)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    logger: BaseLogger = None # Initialize upon instantiation
    per_sample_batch_size: int = field(default=128)
    num_samples: int = field(default=2048)
    batch_id: int = field(default=0) # start from the zero
    
    # guidance related
    guidance_name: str = field(default='no')
    guider: str = field(default='classifier')
    target: str = field(default=None)
    recur_steps: int = field(default=1)
    iter_steps: int = field(default=1)
    guidance_strength: float = field(default=1.0)
    
    # specific for our method (TFG)
    rho: float = field(default=1.0)
    mu: float = field(default=1.0)
    sigma: float = field(default=0.01)
    eps_bsz: int = field(default=4)
    rho_schedule: str = field(default='increase')
    mu_schedule: str = field(default='increase')
    sigma_schedule: str = field(default='decrease')
    clip_scale: float = field(default=1.0)  # ADDED: Missing clip_scale parameter
    
    # cond_fn related
    guide_network: str = field(default='aaraki/vit-base-patch16-224-in21k-finetuned-cifar10')
    classifier_image_size: int = field(default=224)
    
    # evaluation related
    eval_batch_size: int = field(default=128)
    
    # logging related
    logging_resolution: int = field(default=64)
    log_suffix: str = field(default='')
    log_traj: bool = field(default=False)
    max_show_images: int = field(default=256)
    check_done: bool = field(default=True)
    
    # wandb related
    wandb: bool = field(default=False)
    wandb_project: str = field(default='trail')
    wandb_name: str = field(default=None)
    wandb_entity: str = field(default='llm-selection')
    
    # visualization related
    saved_file: str = field(default=None)
    sort_metric: str = field(default=None)
    topk: int = field(default=5)
    output_path: str = field(default='vis_molecule')

    max_n_samples: int = field(default=10000000000)
    
    # bon (best-of-n) related
    bon_rate: float = field(default=1)
    bon_guidance: str = field(default=None)
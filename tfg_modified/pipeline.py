'''
import torch
import os
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod
from diffusion.base import BaseSampler
from methods.base import BaseGuidance
from evaluations.base import BaseEvaluator
from utils.configs import Arguments
import logger

class BasePipeline(object):
    def __init__(self,
                 args: Arguments, 
                 network: BaseSampler, 
                 guider: BaseGuidance, 
                 evaluator: BaseEvaluator,
                 bon_guider=None):
        self.network = network
        self.guider = guider
        self.evaluator = evaluator
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done
        
        self.bon_rate = args.bon_rate
        self.batch_size = args.eval_batch_size
        
        # 初始化 logp_guider，如果没有提供则使用默认的 guider
        self.bon_guider = bon_guider if bon_guider is not None else self.guider
        
    @abstractmethod
    def sample(self, sample_size: int):
        
        samples = self.check_done_and_load_sample()
        
        if samples is None:

            guidance_batch_size = self.batch_size  

            samples = self.network.sample(sample_size=sample_size * self.bon_rate, guidance=self.guider)

            logp_list = []
            for i in range(0, samples.shape[0], guidance_batch_size):
                batch_samples = samples[i:i + guidance_batch_size]
                batch_logp = self.bon_guider.guider.get_guidance(batch_samples, return_logp=True, check_grad=False)
                logp_list.append(batch_logp)

            logp = torch.cat(logp_list, dim=0).view(-1)

            samples = samples.view(sample_size, int(self.bon_rate), *samples.shape[1:])
            logp = logp.view(sample_size, int(self.bon_rate))

            idx = logp.argmax(dim=1)
            samples = samples[torch.arange(sample_size), idx]

            samples = self.network.tensor_to_obj(samples)
                    
        return samples
    
    def evaluate(self, samples):
        return self.check_done_and_evaluate(samples)
    
    def check_done_and_evaluate(self, samples):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, 'metrics.json')):
            logger.log("Metrics already generated. To regenerate, please set `check_done` to `False`.")
            return None
        return self.evaluator.evaluate(samples)

    def check_done_and_load_sample(self):
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, "finished_sampling")):
            logger.log("found tags for generated samples, should load directly. To regenerate, please set `check_done` to `False`.")
            return logger.load_samples()

        return None
'''
import torch
import os
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod
from diffusion.base import BaseSampler
from methods.base import BaseGuidance
from evaluations.base import BaseEvaluator
from utils.configs import Arguments
import logger

class BasePipeline(object):
    def __init__(self,
                 args: Arguments,
                 network: BaseSampler,
                 guider: BaseGuidance,
                 evaluator: BaseEvaluator,
                 bon_guider=None):
        self.network = network
        self.guider = guider
        self.evaluator = evaluator
        self.logging_dir = args.logging_dir
        self.check_done = args.check_done
        self.bon_rate = getattr(args, 'bon_rate', 1)  # Default to 1 if not specified
        self.batch_size = args.eval_batch_size
        
        # Initialize logp_guider
        self.bon_guider = bon_guider if bon_guider is not None else self.guider
        
        print(f"Pipeline initialized:")
        print(f"  - Network: {type(self.network).__name__}")
        print(f"  - Guider: {type(self.guider).__name__}")
        print(f"  - Evaluator: {type(self.evaluator).__name__ if self.evaluator else 'None'}")

    def sample(self, sample_size: int):
        """Generate samples using the configured guidance method"""
        samples = self.check_done_and_load_sample()
        if samples is not None:
            return samples
        
        print(f"Generating {sample_size} samples...")
        guidance_batch_size = self.batch_size
        
        # Generate samples with guidance
        total_samples = sample_size * self.bon_rate
        samples = self.network.sample(sample_size=int(total_samples), guidance=self.guider)
        
        # Apply best-of-n selection if bon_rate > 1
        if self.bon_rate > 1:
            print(f"Applying best-of-n selection (rate: {self.bon_rate})...")
            logp_list = []
            
            for i in range(0, samples.shape[0], guidance_batch_size):
                batch_samples = samples[i:i + guidance_batch_size]
                
                # Handle different guidance interfaces
                try:
                    if hasattr(self.bon_guider, 'guider'):
                        batch_logp = self.bon_guider.guider.get_guidance(
                            batch_samples, return_logp=True, check_grad=False
                        )
                    else:
                        batch_logp = self.bon_guider.get_guidance(
                            batch_samples, return_logp=True, check_grad=False
                        )
                    logp_list.append(batch_logp)
                except Exception as e:
                    print(f"Warning: Failed to compute log probabilities: {e}")
                    # Fallback to random selection
                    batch_logp = torch.randn(batch_samples.shape[0], device=batch_samples.device)
                    logp_list.append(batch_logp)
            
            logp = torch.cat(logp_list, dim=0).view(-1)
            samples = samples.view(sample_size, int(self.bon_rate), *samples.shape[1:])
            logp = logp.view(sample_size, int(self.bon_rate))
            idx = logp.argmax(dim=1)
            samples = samples[torch.arange(sample_size), idx]
        
        # Convert tensors to appropriate format
        samples = self.network.tensor_to_obj(samples)
        print(f"Successfully generated {len(samples)} samples")
        return samples

    def evaluate(self, samples):
        """Evaluate generated samples"""
        return self.check_done_and_evaluate(samples)

    def check_done_and_evaluate(self, samples):
        """Check if evaluation already exists, otherwise run evaluation"""
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, 'metrics.json')):
            logger.log("Metrics already generated. To regenerate, please set `check_done` to `False`.")
            return None
        
        if self.evaluator is None:
            logger.log("No evaluator configured, skipping evaluation.")
            return None
            
        return self.evaluator.evaluate(samples)

    def check_done_and_load_sample(self):
        """Check if samples already exist and load them if so"""
        if self.check_done and os.path.exists(os.path.join(self.logging_dir, "finished_sampling")):
            logger.log("Found existing samples, loading directly. To regenerate, set `check_done` to `False`.")
            return logger.load_samples()
        return None
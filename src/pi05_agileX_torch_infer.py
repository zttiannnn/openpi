from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np
import time
import torch

torch._dynamo.config.suppress_errors = True

model_name = "pi05_agileX"

print(f'Config [{model_name}]....')
config = _config.get_config(model_name)
# Update config to match the converted checkpoint if necessary
# The user converted with action_dim=14, so we should ensure the config reflects that if it doesn't already.
# However, get_config returns the default config. 
# The policy loading mechanism might override some things from the checkpoint config.json, 
# but let's point to the checkpoint directory first.

checkpoint_dir = "/workspace/robot_repo/openpi_torch/openpi/checkpoints/1128_pi05_test_torch/10000"
print(f'Load {model_name} done.')

def _random_observation_agilex() -> dict:
    # AgileX expects a dictionary with "images" and "state" keys, matching AgileXInputs expectations.
    # AgileXInputs expects camera0-camera3 and depth images, which it then converts to base_rgb, feng_rgb, bao_rgb etc.
    # See agileX_policy.py EXPECTED_CAMERAS: ("camera0", "camera1", "camera2", "camera3", "camera0_depth"...)
    
    return {
        # "images": {
        #     "camera0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        #     "camera1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        #     "camera2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        #     "camera3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # },
        "images": {
            "camera0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "camera1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "camera2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "camera3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            # Add depth images (optional, but expected by EXPECTED_CAMERAS)
            "camera0_depth": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "camera1_depth": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "camera2_depth": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "camera3_depth": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        # Assuming 14-dim state as per user's conversion action_dim=14
        "state": np.random.rand(14).astype(np.float32), 
        "prompt": "Do something.",
    }

print('Generating example observation...')
example = _random_observation_agilex()

print('Creating trained policy....')
# We need to make sure the config used here matches the one used for conversion/training.
# The user used --action_dim 14 during conversion.
# We might need to patch the config object if the default pi05_agileX has action_dim=7.
if config.model.action_dim != 14:
    print(f"Updating config action_dim from {config.model.action_dim} to 14 to match checkpoint.")
    # We need to create a new config with updated action_dim
    import dataclasses
    from openpi.models import pi0_config
    
    new_model_config = dataclasses.replace(config.model, action_dim=14)
    config = dataclasses.replace(config, model=new_model_config)

policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Patch the normalization stats in the policy if needed
# The error is that norm_stats has shape (7,) but input state is (14,)
# We need to find the Normalize transform and update its stats.
from openpi import transforms
from openpi.policies import agileX_policy

def patch_norm_stats(policy):
    for transform in policy._input_transform.transforms:
        if isinstance(transform, transforms.Normalize):
            print("Found Normalize transform, checking stats...")
            if transform.norm_stats is not None:
                # The key in norm_stats is "state", not "observation.state"
                state_key = None
                if "state" in transform.norm_stats:
                    state_key = "state"
                elif "observation.state" in transform.norm_stats:
                    state_key = "observation.state"
                
                if state_key is not None:
                    stats = transform.norm_stats[state_key]
                    current_dim = stats.q01.shape[-1] if stats.q01 is not None else (stats.mean.shape[-1] if stats.mean is not None else 0)
                    print(f"  {state_key} norm stats current dim: {current_dim}")
                    
                    if current_dim == 7:
                        print(f"  Patching {state_key} norm stats from 7 to 14 dims...")
                        
                        def pad_stat(arr):
                            if arr is None: 
                                return None
                            arr = np.asarray(arr)
                            # arr is (7,), pad by repeating to make (14,)
                            padding = np.tile(arr, (2,))[:14]  # Double and take first 14
                            return padding

                        new_stats = transforms.NormStats(
                            mean=pad_stat(stats.mean),
                            std=pad_stat(stats.std),
                            q01=pad_stat(stats.q01),
                            q99=pad_stat(stats.q99),
                        )
                        transform.norm_stats[state_key] = new_stats
                        print(f"  Patched {state_key} stats to dim {new_stats.mean.shape[-1] if new_stats.mean is not None else 'N/A'}.")
                else:
                    print("  No 'state' or 'observation.state' key found in norm_stats.")
                    print(f"  Available keys: {list(transform.norm_stats.keys())}")

def patch_use_images(policy):
    """Patch AgileXInputs to enable image processing if needed."""
    for transform in policy._input_transform.transforms:
        if isinstance(transform, agileX_policy.AgileXInputs):
            if not transform.use_images:
                print(f"Found AgileXInputs with use_images={transform.use_images}, patching to True...")
                # AgileXInputs is a frozen dataclass, so we need to use object.__setattr__
                object.__setattr__(transform, 'use_images', True)
                print(f"  Patched AgileXInputs.use_images to {transform.use_images}")

patch_norm_stats(policy)
patch_use_images(policy)

print('Warmup inference...')
action_chunk = policy.infer(example)["actions"]      # 预热模型避免造成统计偏差

print('-' * 50)

inference_count = 10
total_inference_time = 0.0
print('Inference...')
for i in range(inference_count):
    print('-' * 50)
    print(f"Ready to {i+1}/{inference_count} inference...")
    start_time = time.time()
    action_chunk = policy.infer(example)["actions"]
    end_time = time.time()
    print(f'Inference done, cost time {end_time - start_time:.3f} s')
    print(f"Action chunk shape: {action_chunk.shape}")
    # print(action_chunk) # Optional: print actions
    total_inference_time += (end_time - start_time)
    
print(f'Total inference done, average cost time: {(total_inference_time / inference_count)} s')

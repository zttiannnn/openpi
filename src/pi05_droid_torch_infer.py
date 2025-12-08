from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import time

import torch
torch._dynamo.config.suppress_errors = True

model_name = "pi05_droid"

print(f'Config [{model_name}]....')
config = _config.get_config(model_name)
checkpoint_dir = "./torch_pi05_droid"
print(f'Load {model_name} done.')

def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

print('Generating example observation...')
example = _random_observation_droid()

print('Creating trained policy....')
policy = policy_config.create_trained_policy(config, checkpoint_dir)
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
    print(action_chunk)
    total_inference_time += (end_time - start_time)
    
print(f'Total inference done, average cost time: {(total_inference_time / inference_count)} s')

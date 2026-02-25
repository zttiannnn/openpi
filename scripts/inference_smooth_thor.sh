python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/thor_torch_pi05_agileX/57500/ \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953001CP, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 20 \
  --fps 20 \
  --mode pose
python scripts/evaluate.py run \
 --config pi05_agileX_thor \
 --checkpoint_dir ./checkpoints/pi05_agileX_thor/grad_acc_4_lr_2e-4_1e-6/20000 \
 --repo_id lerobot/test \
 --root /workspace/JE_robot_data_lerobot/0228_data_lerobot \
 --episode_id 1 \
 --default_prompt "Put the purple carton of milk into the cardboard box." \
 --out ./thor_0309_20000_data0228.npz \
 --plot-after-run \
 --out-png ./thor_0309_20000_data0228.png
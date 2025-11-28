#!/bin/bash

#MIOPEN_USE_GEMM=1 \
#MIOPEN_DISABLE_ASM_KERNELS=1 \
#MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM=0 \
#MIOPEN_DISABLE_BATCH_NORM_ASM_KERNELS=1 \
#MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING=0 \
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
python -m ipdb -c c ./ppo_etd/train.py \
    --exp_name="test_count_kcs6r3_nep4_bnorm_ent1e-2_extR1_intR1e-2+SEED=10" \
    --game_name="KeyCorridorS6R3" \
    --run_id=10 \
    --int_rew_source="CountFirstVisit" \
    --count_feedbacks_type="normal" \
    --env_source="minigrid" \
    --use_wandb=0 \
    --int_rew_coef=1e-2 \
    --ext_rew_coef=1.0 \
    --ent_coef=1e-2 \
    --max_grad_norm=0.5 \
    --offpolicy_data=0 \
    --n_epochs=4 \
    --use_model_rnn=0 \
    --record_video=0 \
    --enable_plotting=0 \
    --model_mlp_layers=1 \
    --log_explored_states=1 \
    --total_steps=5_000_000 \
    --image_noise_scale=0 \
    --batch_size=512 \
    --discount=0.99 \
    --policy_cnn_type=0 \
    --model_cnn_type=0 \
    --n_steps=512 \
    --num_processes=2 \
    --features_dim=64 \
    --model_features_dim=64 \
    --model_cnn_norm=BatchNorm \
    --model_mlp_norm=BatchNorm \
    --policy_cnn_norm=BatchNorm \
    --policy_mlp_norm=BatchNorm \
 
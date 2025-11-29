#!/bin/bash

#MIOPEN_USE_GEMM=1 \
#MIOPEN_DISABLE_ASM_KERNELS=1 \
#MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM=0 \
#MIOPEN_DISABLE_BATCH_NORM_ASM_KERNELS=1 \
#MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING=0 \
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python ./ppo_etd/train.py \
    --game_name="FullMazeS5" \
    --run_id=40 \
    --int_rew_source="TDD" \
    --env_source=miniworld \
    --exp_name="test_tdd_mazeS5_nmep16_fdim512_lnorm_mlp1_ent1e-2_extR10_mlr1e-4+SEED=40" \
    --use_wandb=1 \
    --project_name="MiniWorld-TDD-FullMazeS5" \
    --int_rew_coef=1e-2 \
    --ext_rew_coef=10.0 \
    --model_learning_rate=1e-4 \
    --ent_coef=1e-2 \
    --max_grad_norm=0.5 \
    --tdd_aggregate_fn=min \
    --model_n_epochs=16 \
    --use_model_rnn=0 \
    --record_video=0 \
    --enable_plotting=0 \
    --model_mlp_layers=1 \
    --total_steps=5_000_000 \
    --image_noise_scale=0 \
    --batch_size=512 \
    --discount=0.99 \
    --policy_cnn_type=1 \
    --model_cnn_type=1 \
    --n_steps=512 \
    --num_processes=16 \
    --features_dim=512 \
    --model_features_dim=512 \
    --model_cnn_norm=LayerNorm \
    --model_mlp_norm=LayerNorm \
    --policy_cnn_norm=LayerNorm \
    --policy_mlp_norm=LayerNorm 


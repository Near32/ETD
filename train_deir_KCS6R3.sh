#!/bin/bash

#MIOPEN_USE_GEMM=1 \
#MIOPEN_DISABLE_ASM_KERNELS=1 \
#MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM=0 \
#MIOPEN_DISABLE_BATCH_NORM_ASM_KERNELS=1 \
#MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING=0 \
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python ./ppo_etd/train.py \
    --game_name="KeyCorridorS6R3" \
    --run_id=20 \
    --int_rew_source="DEIR" \
    --env_source=minigrid \
    --exp_name="test_deir_kcs6r3_nproc16_nmep4_rnn1_bnorm_mlp1_ent1e-2_extR1+intR1e-2_mlr3e-4+ObsQueue=1e5+SEED=20" \
    --use_wandb=1 \
    --project_name="MiniWorld-DEIR-KeyCorridorS6R3" \
    --int_rew_coef=1e-2 \
    --ext_rew_coef=1.0 \
    --model_learning_rate=3e-4 \
    --ent_coef=1e-2 \
    --dsc_obs_queue_len=10_000 \
    --max_grad_norm=0.5 \
    --tdd_aggregate_fn=min \
    --model_n_epochs=4 \
    --n_epochs=4 \
    --use_model_rnn=1 \
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
    --num_processes=16 \
    --model_cnn_norm=BatchNorm \
    --model_mlp_norm=BatchNorm \
    --policy_cnn_norm=BatchNorm \
    --policy_mlp_norm=BatchNorm 


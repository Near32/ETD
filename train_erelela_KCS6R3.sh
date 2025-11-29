#!/bin/bash

GPU_ID=${1:-0}
RUN_ID=${2:-10}

EXP_NAME=${3:-"erelela_keycorridor_s6r3+ExtR=1.0+IntR=1e-2+RGEp=2+ExprThr=40vs20+CAMResample+RGPeriod=512k+SEED=10"}
GAME_NAME="KeyCorridorS6R3"
PROJECT_NAME="EReLELA-KeyCorridor-S6R3"
#ERELELA_CONFIG="../../IMPALA/RIDE/impala_ride/Regym/benchmark/EReLELA/MiniGrid/keycorridor_S6_R3_minigrid_wandb_benchmark_AgnosticPOMDPERELELA_config.yaml"
#ERELELA_CONFIG="../../IMPALA/RIDE/impala_ride/Regym/benchmark/EReLELA/MiniGrid/keycorridor_S3_R3_symbolic_minigrid_wandb_ETD_benchmark_AgnosticPOMDPERELELA+R2D2_config.yaml"
#ERELELA_CONFIG="../../IMPALA/RIDE/configs/keycorridor_S3_R3_minigrid_wandb_RIDE_benchmark_AgnosticPOMDPERELELA_config.yaml"
#ERELELA_CONFIG="./configs/keycorridor_S3_R3_minigrid_wandb_ETD_benchmark_AgnosticPOMDPERELELA_config.yaml"
ERELELA_CONFIG="./configs/keycorridor_S3_R3_minigrid_wandb_SmallETD_benchmark_AgnosticPOMDPERELELA_config.yaml"

#MIOPEN_DEBUG_DISABLE_FIND_DB=1 \
#MIOPEN_FIND_MODE=NORMAL \
#MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0 \
CUDA_VISIBLE_DEVICES=${GPU_ID} \
PYTHONPATH="./" \
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
python -m ipdb -c c ./ppo_etd/train.py \
    --exp_name ${EXP_NAME} \
    --project_name ${PROJECT_NAME} \
    --game_name=${GAME_NAME} \
    --env_source=minigrid \
    --run_id=${RUN_ID} \
    --use_wandb=1 \
    --int_rew_source=EReLELA \
    --erelela_config=${ERELELA_CONFIG} \
    --erelela_intrinsic_weight=0.01 \
    --erelela_extrinsic_weight=1.0 \
    --erelela_feedbacks_type=normal \
    --total_steps=5000000 \
    --int_rew_coef=0.01 \
    --ent_coef=5e-4 \
    --n_epochs=8 \
    --model_n_epochs=8 \
    --learning_rate=3e-4 \
    --model_learning_rate=1e-6 \
    --max_grad_norm=0.5 \
    --num_processes=16 \
    --discount=0.99 \
    --model_mlp_layers=1 \
    --n_steps=512 \
    --batch_size=512 \
    --features_dim=64 \
    --model_features_dim=64 \
    --policy_cnn_type=0 \
    --model_cnn_type=0 \
    --policy_mlp_norm=BatchNorm \
    --model_mlp_norm=BatchNorm \
    --policy_cnn_norm=BatchNorm \
    --model_cnn_norm=BatchNorm \
    --record_video=0 \
    --enable_plotting=0 \
    --use_baseline_ther_wrapper=1 \
    --erelela_override=success_threshold=0.01 \
    --erelela_override=add_rgb_wrapper=False \
    --erelela_override=language_guided_curiosity=False \
    --erelela_override=language_guided_curiosity_descr_type=descr \
    --erelela_override=language_guided_curiosity_extrinsic_weight=10.0 \
    --erelela_override=language_guided_curiosity_intrinsic_weight=0.1 \
    --erelela_override=language_guided_curiosity_binary_reward=False \
    --erelela_override=language_guided_curiosity_densify=False \
    --erelela_override=language_guided_curiosity_non_episodic_dampening_rate=0.0 \
    --erelela_override=coverage_manipulation_metric=True \
    --erelela_override=MiniWorld_entity_visibility_oracle=False \
    --erelela_override=MiniWorld_entity_visibility_oracle_language_specs=none \
    --erelela_override=MiniWorld_entity_visibility_oracle_too_far_threshold=-1.0 \
    --erelela_override=MiniWorld_entity_visibility_oracle_include_discrete_depth=True \
    --erelela_override=MiniWorld_entity_visibility_oracle_include_depth_precision=-1.0 \
    --erelela_override=MiniWorld_entity_visibility_oracle_top_view=False \
    --erelela_override=PER_alpha=0.5 \
    --erelela_override=PER_beta=1.0 \
    --erelela_override=PER_use_rewards_in_priority=False \
    --erelela_override=sequence_replay_PER_eta=0.9 \
    --erelela_override=PER_compute_initial_priority=True \
    --erelela_override=use_ELA=True \
    --erelela_override=ELA_use_ELA=True \
    --erelela_override=use_HER=False \
    --erelela_override=goal_oriented=False \
    --erelela_override=ELA_with_rg_training=True \
    --erelela_override=ELA_with_rg_optimize=True \
    --erelela_override=ELA_rg_use_cuda=True \
    --erelela_override=ELA_rg_dataloader_shuffle=True \
    --erelela_override=ELA_rg_dataloader_num_worker=4 \
    --erelela_override=ELA_rg_graphtype=straight_through_gumbel_softmax \
    --erelela_override=ELA_rg_obverter_threshold_to_stop_message_generation=0.9 \
    --erelela_override=ELA_rg_obverter_nbr_games_per_round=32 \
    --erelela_override=ELA_rg_obverter_sampling_round_alternation_only=False \
    --erelela_override=ELA_rg_use_obverter_sampling=False \
    --erelela_override=ELA_rg_compactness_ambiguity_metric_language_specs=emergent+natural+color+shape+shuffled-emergent+shuffled-natural+shuffled-color+shuffled-shape \
    --erelela_override=ELA_rg_compactness_ambiguity_metric_resampling=True \
    --erelela_override=ELA_rg_sanity_check_compactness_ambiguity_metric=False \
    --erelela_override=ELA_rg_shared_architecture=True \
    --erelela_override=ELA_rg_with_logits_mdl_principle=False \
    --erelela_override=ELA_rg_logits_mdl_principle_factor=0.0 \
    --erelela_override=ELA_rg_logits_mdl_principle_accuracy_threshold=60.0 \
    --erelela_override=ELA_rg_agent_loss_type=Impatient+Hinge \
    --erelela_override=ELA_rg_use_semantic_cooccurrence_grounding=False \
    --erelela_override=ELA_rg_semantic_cooccurrence_grounding_lambda=1.0 \
    --erelela_override=ELA_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
    --erelela_override=ELA_lock_test_storage=False \
    --erelela_override=ELA_rg_color_jitter_prob=0.0 \
    --erelela_override=ELA_rg_gaussian_blur_prob=0.5 \
    --erelela_override=ELA_rg_egocentric_prob=0.0 \
    --erelela_override=ELA_rg_object_centric_version=2 \
    --erelela_override=ELA_rg_descriptive_version=1 \
    --erelela_override=ELA_rg_learning_rate=3e-4 \
    --erelela_override=ELA_rg_weight_decay=0.0 \
    --erelela_override=ELA_rg_l1_weight_decay=0.0 \
    --erelela_override=ELA_rg_l2_weight_decay=0.0 \
    --erelela_override=ELA_rg_vocab_size=64 \
    --erelela_override=ELA_rg_max_sentence_length=128 \
    --erelela_override=ELA_rg_training_period=512000 \
    --erelela_override=ELA_rg_training_max_skip=32 \
    --erelela_override=ELA_rg_training_adaptive_period=False \
    --erelela_override=ELA_rg_descriptive=True \
    --erelela_override=ELA_rg_use_curriculum_nbr_distractors=False \
    --erelela_override=ELA_rg_nbr_epoch_per_update=2 \
    --erelela_override=ELA_rg_accuracy_threshold=90 \
    --erelela_override=ELA_rg_relative_expressivity_threshold=90 \
    --erelela_override=ELA_rg_expressivity_threshold=40 \
    --erelela_override=ELA_rg_nbr_train_distractors=256 \
    --erelela_override=ELA_rg_nbr_test_distractors=3 \
    --erelela_override=ELA_replay_capacity=8192 \
    --erelela_override=ELA_test_replay_capacity=2048 \
    --erelela_override=ELA_rg_distractor_sampling_scheme_version=2 \
    --erelela_override=ELA_rg_distractor_sampling_with_replacement=True \
    --erelela_override=ELA_rg_same_episode_target=True \
    --erelela_override=ELA_rg_distractor_sampling=uniform \
    --erelela_override=ELA_feedbacks_failure_reward=0.0 \
    --erelela_override=ELA_feedbacks_success_reward=1 \
    --erelela_override=BabyAI_Bot_action_override=False \
    --erelela_override=n_step=3 \
    --erelela_override=nbr_actor=32 \
    --erelela_override=epsstart=1.0 \
    --erelela_override=epsend=0.1 \
    --erelela_override=epsdecay=1000000 \
    --erelela_override=eps_greedy_alpha=2.0 \
    --erelela_override=nbr_minibatches=1 \
    --erelela_override=batch_size=32 \
    --erelela_override=min_capacity=4e3 \
    --erelela_override=min_handled_experiences=2.8e4 \
    --erelela_override=replay_capacity=2.0e4 \
    --erelela_override=learning_rate=1e-4 \
    --erelela_override=sequence_replay_burn_in_ratio=0.5 \
    --erelela_override=weights_entropy_lambda=0.0 \
    --erelela_override=sequence_replay_unroll_length=20 \
    --erelela_override=sequence_replay_overlap_length=10 \
    --erelela_override=sequence_replay_use_online_states=True \
    --erelela_override=sequence_replay_use_zero_initial_states=False \
    --erelela_override=sequence_replay_store_on_terminal=False \
    --erelela_override=HER_target_clamping=False \
    --erelela_override=adam_weight_decay=0.0 \
    --erelela_override=ther_adam_weight_decay=0.0 \
    --erelela_override=nbr_training_iteration_per_cycle=2 \
    --erelela_override=nbr_episode_per_cycle=0 \
    --erelela_override=single_pick_episode=False \
    --erelela_override=terminate_on_completion=True \
    --erelela_override=allow_carrying=False \
    --erelela_override=time_limit=0 \
    --erelela_override=benchmarking_record_episode_interval=4 \
    --erelela_override=benchmarking_interval=1.0e4 \
    --erelela_override=train_observation_budget=1.0e7

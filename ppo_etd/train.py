# Disable Warning
import logging
import copy
import time
logging.captureWarnings(True)
import os
os.environ["CUDNN_LOGINFO_DBG"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import ast
import click
import warnings
import torch as th
th.backends.cudnn.enabled = False
th.backends.mkldnn.enabled = True
if th.cuda.is_available():
    th.cuda.init()
    _ = th.zeros(1, device='cuda')  # Force full init
else:
    raise NotImplementedError

# noinspection PyUnresolvedReferences
#try:
#    import gym_miniworld
#    #import miniworld
#except Exception as e:
#    pass

from ppo_etd.env.minigrid_envs import *
from ppo_etd.algo.ppo_model import PPOModel
from ppo_etd.algo.ppo_trainer import PPOTrainer
from ppo_etd.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed

warnings.filterwarnings("ignore", category=DeprecationWarning)


ELA_BOOL_KEYS = {
    'use_ELA',
    'ELA_use_ELA',
    'ELA_with_rg_training',
    'ELA_with_rg_optimize',
    'ELA_rg_dataloader_shuffle',
    'ELA_rg_compactness_ambiguity_metric_with_ordering',
    'ELA_rg_compactness_ambiguity_metric_use_cumulative_scores',
    'ELA_rg_compactness_ambiguity_metric_resampling',
    'ELA_rg_compactness_ambiguity_metric_resample_progress',
    'ELA_rg_sanity_check_compactness_ambiguity_metric',
    'ELA_rg_training_adaptive_period',
    'ELA_rg_verbose',
    'ELA_rg_use_cuda',
    'ELA_rg_with_semantic_grounding_metric',
    'ELA_rg_use_semantic_cooccurrence_grounding',
    'ELA_rg_record_unique_stats',
    'ELA_rg_filter_out_non_unique',
    'ELA_lock_test_storage',
    'ELA_rg_same_episode_target',
    'ELA_rg_with_color_jitter_augmentation',
    'ELA_rg_with_gaussian_blur_augmentation',
    'ELA_rg_egocentric',
    'ELA_rg_descriptive',
    'ELA_rg_distractor_sampling_with_replacement',
    'ELA_rg_object_centric',
    'ELA_rg_force_eos',
    'ELA_rg_shared_architecture',
    'ELA_rg_normalize_features',
    'ELA_rg_with_logits_mdl_principle',
    'ELA_rg_logits_mdl_principle_normalization',
    'ELA_rg_logits_mdl_principle_use_inst_accuracy',
    'ELA_rg_iterated_learning_scheme',
    'ELA_rg_iterated_learning_rehearse_MDL',
    'ELA_rg_use_obverter_sampling',
    'ELA_rg_obverter_sampling_round_alternation_only',
    'ELA_rg_homoscedastic_multitasks_loss',
    'ELA_rg_use_feat_converter',
    'ELA_rg_use_curriculum_nbr_distractors',
    'ELA_rg_metric_fast',
    'ELA_rg_metric_resampling',
    'ELA_rg_dis_metric_resampling',
    'ELA_rg_metric_active_factors_only',
}

DEFAULT_ERELELA_OVERRIDES = {
    'use_ELA': False,
    'ELA_use_ELA': False,
    'ELA_with_rg_training': True,
    'ELA_with_rg_optimize': True,
    'ELA_reward_extrinsic_weight': 1.0,
    'ELA_reward_intrinsic_weight': 1.0,
    'ELA_feedbacks_type': 'normal',
    'ELA_feedbacks_failure_reward': 0.0,
    'ELA_feedbacks_success_reward': 1.0,
    'ELA_rg_dataloader_shuffle': True,
    'ELA_rg_language_dynamic_metric_epoch_period': 32,
    'ELA_rg_compactness_ambiguity_metric_epoch_period': 1,
    'ELA_rg_compactness_ambiguity_metric_with_ordering': False,
    'ELA_rg_compactness_ambiguity_metric_use_cumulative_scores': True,
    'ELA_rg_compactness_ambiguity_metric_language_specs': 'emergent',
    'ELA_rg_compactness_ambiguity_metric_resampling': False,
    'ELA_rg_compactness_ambiguity_metric_resample_batch_size': 64,
    'ELA_rg_compactness_ambiguity_metric_resample_progress': False,
    'ELA_rg_sanity_check_compactness_ambiguity_metric': False,
    'ELA_rg_training_period': 1024,
    'ELA_rg_training_max_skip': -1,
    'ELA_rg_training_adaptive_period': False,
    'ELA_rg_accuracy_threshold': 75.0,
    'ELA_rg_relative_expressivity_threshold': 0.0,
    'ELA_rg_expressivity_threshold': 0.0,
    'ELA_rg_verbose': True,
    'ELA_rg_use_cuda': False,
    'ELA_exp_key': 'succ_s',
    'ELA_rg_with_semantic_grounding_metric': False,
    'ELA_rg_use_semantic_cooccurrence_grounding': False,
    'ELA_grounding_signal_key': 'info:desired_goal',
    'ELA_rg_semantic_cooccurrence_grounding_lambda': 1.0,
    'ELA_rg_semantic_cooccurrence_grounding_noise_magnitude': 0.0,
    'ELA_split_strategy': 'divider-1-offset-0',
    'ELA_rg_record_unique_stats': False,
    'ELA_rg_filter_out_non_unique': False,
    'ELA_replay_capacity': 1024,
    'ELA_lock_test_storage': False,
    'ELA_rg_same_episode_target': False,
    'ELA_test_replay_capacity': 512,
    'ELA_test_train_split_interval': 5,
    'ELA_train_dataset_length': None,
    'ELA_test_dataset_length': None,
    'ELA_rg_object_centric_version': 1,
    'ELA_rg_descriptive_version': 2,
    'ELA_rg_with_color_jitter_augmentation': False,
    'ELA_rg_color_jitter_prob': 0.0,
    'ELA_rg_with_gaussian_blur_augmentation': False,
    'ELA_rg_gaussian_blur_prob': 0.0,
    'ELA_rg_egocentric_tr_degrees': 15.0,
    'ELA_rg_egocentric_tr_xy': 10.0,
    'ELA_rg_egocentric': False,
    'ELA_rg_egocentric_prob': 0.0,
    'ELA_rg_nbr_train_distractors': 7,
    'ELA_rg_nbr_test_distractors': 7,
    'ELA_rg_descriptive': False,
    'ELA_rg_descriptive_ratio': 0.0,
    'ELA_rg_observability': 'partial',
    'ELA_rg_max_sentence_length': 10,
    'ELA_rg_distractor_sampling': 'uniform',
    'ELA_rg_distractor_sampling_scheme_version': 1,
    'ELA_rg_distractor_sampling_with_replacement': False,
    'ELA_rg_object_centric': False,
    'ELA_rg_object_centric_type': 'hard',
    'ELA_rg_graphtype': 'straight_through_gumbel_softmax',
    'ELA_rg_vocab_size': 32,
    'ELA_rg_force_eos': True,
    'ELA_rg_symbol_embedding_size': 64,
    'ELA_rg_arch': 'BN+7x4x3xCNN',
    'ELA_rg_shared_architecture': False,
    'ELA_rg_normalize_features': False,
    'ELA_rg_agent_loss_type': 'Hinge',
    'ELA_rg_with_logits_mdl_principle': False,
    'ELA_rg_logits_mdl_principle_normalization': False,
    'ELA_rg_logits_mdl_principle_use_inst_accuracy': False,
    'ELA_rg_logits_mdl_principle_factor': 1.0e-3,
    'ELA_rg_logits_mdl_principle_accuracy_threshold': 10.0,
    'ELA_rg_cultural_pressure_it_period': 0,
    'ELA_rg_cultural_speaker_substrate_size': 1,
    'ELA_rg_cultural_listener_substrate_size': 1,
    'ELA_rg_cultural_reset_strategy': 'uniformSL',
    'ELA_rg_cultural_pressure_meta_learning_rate': 1.0e-3,
    'ELA_rg_iterated_learning_scheme': False,
    'ELA_rg_iterated_learning_period': 5,
    'ELA_rg_iterated_learning_rehearse_MDL': False,
    'ELA_rg_iterated_learning_rehearse_MDL_factor': 1.0,
    'ELA_rg_obverter_threshold_to_stop_message_generation': 0.9,
    'ELA_rg_obverter_nbr_games_per_round': 20,
    'ELA_rg_use_obverter_sampling': False,
    'ELA_rg_obverter_sampling_round_alternation_only': False,
    'ELA_rg_batch_size': 32,
    'ELA_rg_dataloader_num_worker': 8,
    'ELA_rg_learning_rate': 3.0e-4,
    'ELA_rg_weight_decay': 0.0,
    'ELA_rg_l1_weight_decay': 0.0,
    'ELA_rg_l2_weight_decay': 0.0,
    'ELA_rg_dropout_prob': 0.0,
    'ELA_rg_emb_dropout_prob': 0.0,
    'ELA_rg_homoscedastic_multitasks_loss': False,
    'ELA_rg_use_feat_converter': True,
    'ELA_rg_use_curriculum_nbr_distractors': False,
    'ELA_rg_init_curriculum_nbr_distractors': 1,
    'ELA_rg_nbr_experience_repetition': 1,
    'ELA_rg_agent_nbr_latent_dim': 32,
    'ELA_rg_symbol_processing_nbr_hidden_units': 512,
    'ELA_rg_mini_batch_size': 32,
    'ELA_rg_optimizer_type': 'adam',
    'ELA_rg_nbr_epoch_per_update': 3,
    'ELA_rg_metric_epoch_period': 10024,
    'ELA_rg_dis_metric_epoch_period': 10024,
    'ELA_rg_metric_batch_size': 16,
    'ELA_rg_metric_fast': True,
    'ELA_rg_parallel_TS_worker': 8,
    'ELA_rg_nbr_train_points': 1024,
    'ELA_rg_nbr_eval_points': 512,
    'ELA_rg_metric_resampling': True,
    'ELA_rg_dis_metric_resampling': True,
    'ELA_rg_seed': 1,
    'ELA_rg_metric_active_factors_only': True,
    'THER_observe_achieved_goal': False,
    'MiniWorld_symbolic_image': False,
}


def _extract_scalar_items(source_dict):
    scalars = {}
    if not isinstance(source_dict, dict):
        return scalars
    for key, value in source_dict.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            scalars[key] = value
        elif isinstance(value, (list, tuple, dict)):
            scalars[key] = copy.deepcopy(value)
    return scalars


def _load_erelela_config_scalars(config_path):
    if not config_path:
        return {}
    try:
        from impala_ride.algos.ela import load_configs
        _, agents_cfg, tasks_cfg = load_configs(config_path, kwargs={})
    except Exception as exc:
        logging.warning("Could not preload EReLELA config '%s': %s", config_path, exc)
        return {}

    if not tasks_cfg:
        return {}

    task_cfg = tasks_cfg[0]
    agent_cfg = agents_cfg.get(task_cfg.get('agent-id'), {}) if isinstance(agents_cfg, dict) else {}

    scalars = {}
    scalars.update(_extract_scalar_items(task_cfg))
    scalars.update(_extract_scalar_items(agent_cfg))
    return scalars


def parse_key_value_pairs(pairs, erelela_config_path=None, prefill=None):
    result = {}
    if erelela_config_path:
        result.update(_load_erelela_config_scalars(erelela_config_path))
    
    if prefill is not None:
        result.update(prefill) 

    if pairs:
        for entry in pairs:
            if entry is None or '=' not in entry:
                continue
            key, value = entry.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            try:
                parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value
            if "None" in value:
                parsed_value = None
            result[key] = parsed_value

    return result


def train(config):
    th.autograd.set_detect_anomaly(False)
    th.set_default_dtype(th.float32)
    th.backends.cudnn.benchmark = False

    # Make sure baseline wrapper toggles/kwargs are ready before environments are built.
    erelela_overrides = None
    baseline_prefill = dict(DEFAULT_ERELELA_OVERRIDES)
    if config.erelela_config:
        overrides_prefill = dict(DEFAULT_ERELELA_OVERRIDES)
        erelela_overrides = parse_key_value_pairs(
            config.erelela_override,
            erelela_config_path=config.erelela_config,
            prefill=overrides_prefill,
        )
        # Update for specific command line arguments:
        erelela_overrides.update({
            'ELA_reward_intrinsic_weight': config.erelela_intrinsic_weight,
            'ELA_reward_extrinsic_weight': config.erelela_extrinsic_weight,
            'ELA_feedbacks_type': config.erelela_feedbacks_type,
        })

        factor = erelela_overrides["ELA_rg_logits_mdl_principle_factor"]
        if isinstance(factor, str):
            if '-' in factor:
                betas = [float(beta) for beta in factor.split('-')]
                assert len(betas) == 2
            else:
                betas = None 
                factor = float(factor)
        else:
            betas = None 
            factor = float(factor)

        if betas is not None or factor > 0.0:
            erelela_overrides["ELA_rg_with_logits_mdl_principle"] = True
            if betas is not None:
                erelela_overrides["ELA_rg_logits_mdl_principle_accuracy_threshold"] = 0.0

        if 'episodic-dissimilarity' in erelela_overrides['ELA_rg_distractor_sampling']:
            erelela_overrides['ELA_rg_same_episode_target'] = True 

        if erelela_overrides['ELA_rg_gaussian_blur_prob'] > 0.0 :
            erelela_overrides['ELA_rg_with_gaussian_blur_augmentation'] = True

        if erelela_overrides['ELA_rg_color_jitter_prob'] > 0.0 :
            erelela_overrides['ELA_rg_with_color_jitter_augmentation'] = True

        if erelela_overrides['ELA_rg_egocentric_prob'] > 0.0 :
            erelela_overrides['ELA_rg_egocentric'] = True

        if "natural" in erelela_overrides["ELA_rg_compactness_ambiguity_metric_language_specs"]:
            erelela_overrides["THER_observe_achieved_goal"] = True
            print(f"WARNING: ELA_rg_compactness_ambiguity_metric_language_specs contains 'natural'. Thus, THER_observed_achieved_goal is set to True. Necessary for the MultiRoom envs to have BehaviouralDescriptions in NL.")
            time.sleep(1)

        if erelela_overrides['language_guided_curiosity']:
            erelela_overrides['coverage_manipulation_metric'] = True
            if 'descr' not in erelela_overrides['language_guided_curiosity_descr_type']:
                erelela_overrides["MiniWorld_entity_visibility_oracle"] = True
        
        baseline_prefill = dict(erelela_overrides)
        print(erelela_overrides)

    config.use_baseline_ther_wrapper = bool(config.use_baseline_ther_wrapper)
    '''
    config.baseline_ther_kwargs = parse_key_value_pairs(
        config.baseline_ther_arg,
        erelela_config_path=config.erelela_config,
        prefill=baseline_prefill,
    )
    '''
    if config.use_baseline_ther_wrapper:
        config.baseline_ther_kwargs = dict(
            size=baseline_prefill['observation_resize_dim'], 
            skip=baseline_prefill['nbr_frame_skipping'], 
            stack=baseline_prefill['nbr_frame_stacking'],
            single_life_episode=baseline_prefill['single_life_episode'],
            nbr_max_random_steps=baseline_prefill['nbr_max_random_steps'],
            clip_reward=baseline_prefill['clip_reward'],
            time_limit=baseline_prefill['time_limit'],
            max_sentence_length=128, #agent_config['THER_max_sentence_length'] if agent_config['use_THER'] else agent_config['ELA_rg_max_sentence_length'],
            vocabulary=baseline_prefill['THER_vocabulary'],
            vocab_size=baseline_prefill['THER_vocab_size'],
            previous_reward_action=baseline_prefill['previous_reward_action'],
            observation_key=baseline_prefill['observation_key'],
            concatenate_keys_with_obs=baseline_prefill['concatenate_keys_with_obs'],
            add_rgb_wrapper=baseline_prefill['add_rgb_wrapper'],
            full_obs=baseline_prefill['full_obs'],
            single_pick_episode=baseline_prefill['single_pick_episode'],
            observe_achieved_pickup_goal=baseline_prefill['THER_observe_achieved_goal'],
            use_visible_entities=False, #('visible-entities' in task_config['ETHER_with_Oracle_type']),
            babyai_mission=baseline_prefill['BabyAI_Bot_action_override'],
            miniworld_symbolic_image=baseline_prefill['MiniWorld_symbolic_image'],
            miniworld_entity_visibility_oracle=baseline_prefill['MiniWorld_entity_visibility_oracle'],
            miniworld_entity_visibility_oracle_language_specs=baseline_prefill['MiniWorld_entity_visibility_oracle_language_specs'],
            miniworld_entity_visibility_oracle_include_discrete_depth=baseline_prefill['MiniWorld_entity_visibility_oracle_include_discrete_depth'],
            miniworld_entity_visibility_oracle_include_depth=baseline_prefill['MiniWorld_entity_visibility_oracle_include_depth'],
            miniworld_entity_visibility_oracle_include_depth_precision=baseline_prefill['MiniWorld_entity_visibility_oracle_include_depth_precision'],
            miniworld_entity_visibility_oracle_too_far_threshold=baseline_prefill['MiniWorld_entity_visibility_oracle_too_far_threshold'],
            miniworld_entity_visibility_oracle_top_view=baseline_prefill['MiniWorld_entity_visibility_oracle_top_view'],
            language_guided_curiosity=baseline_prefill['language_guided_curiosity'],
            language_guided_curiosity_extrinsic_weight=baseline_prefill['language_guided_curiosity_extrinsic_weight'],
            language_guided_curiosity_intrinsic_weight=baseline_prefill['language_guided_curiosity_intrinsic_weight'],
            language_guided_curiosity_binary_reward=baseline_prefill['language_guided_curiosity_binary_reward'],
            language_guided_curiosity_densify=baseline_prefill['language_guided_curiosity_densify'],
            ne_count_based_exploration=baseline_prefill['language_guided_curiosity_non_episodic_count_based_exploration'],
            ne_dampening_rate=baseline_prefill['language_guided_curiosity_non_episodic_dampening_rate'],
            coverage_manipulation_metric=baseline_prefill['coverage_manipulation_metric'],
            descr_type=baseline_prefill['language_guided_curiosity_descr_type'],
        )

    wrapper_class = config.get_wrapper_class()
    venv = config.get_venv(wrapper_class)
    callbacks = config.get_callbacks()
    optimizer_class, optimizer_kwargs = config.get_optimizer()
    activation_fn, cnn_activation_fn = config.get_activation_fn()
    config.cast_enum_values()
    if config.policy_cnn_type == -1 or config.model_cnn_type == -1: # Observation Space is not image
        policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs = \
            config.get_mlp_kwargs(activation_fn)
    else:
        policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs = \
            config.get_cnn_kwargs(cnn_activation_fn)

    policy_kwargs = dict(
        run_id=config.run_id,
        n_envs=config.num_processes,
        activation_fn=activation_fn,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        policy_features_extractor_class=policy_features_extractor_class,
        policy_features_extractor_kwargs=features_extractor_common_kwargs,
        model_cnn_features_extractor_class=model_cnn_features_extractor_class,
        model_cnn_features_extractor_kwargs=model_features_extractor_common_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=config.max_grad_norm,
        model_features_dim=config.model_features_dim,
        latents_dim=config.latents_dim,
        model_latents_dim=config.model_latents_dim,
        policy_mlp_norm=config.policy_mlp_norm,
        model_mlp_norm=config.model_mlp_norm,
        model_cnn_norm=config.model_cnn_norm,
        model_mlp_layers=config.model_mlp_layers,
        use_status_predictor=config.use_status_predictor,
        gru_layers=config.gru_layers,
        policy_mlp_layers=config.policy_mlp_layers,
        policy_gru_norm=config.policy_gru_norm,
        use_model_rnn=config.use_model_rnn,
        model_gru_norm=config.model_gru_norm,
        total_timesteps=config.total_steps,
        n_steps=config.n_steps,
        int_rew_source=config.int_rew_source,
        icm_forward_loss_coef=config.icm_forward_loss_coef,
        ngu_knn_k=config.ngu_knn_k,
        ngu_dst_momentum=config.ngu_dst_momentum,
        ngu_use_rnd=config.ngu_use_rnd,
        rnd_err_norm=config.rnd_err_norm,
        rnd_err_momentum=config.rnd_err_momentum,
        rnd_use_policy_emb=config.rnd_use_policy_emb,
        dsc_obs_queue_len=config.dsc_obs_queue_len,
        log_dsc_verbose=config.log_dsc_verbose,
        tdd_aggregate_fn=config.tdd_aggregate_fn,
        tdd_energy_fn=config.tdd_energy_fn,
        tdd_loss_fn=config.tdd_loss_fn,
        tdd_logsumexp_coef=config.tdd_logsumexp_coef,
        offpolicy_data=config.offpolicy_data,
        count_feedbacks_type=config.count_feedbacks_type,
    )

    policy_kwargs.update(dict(
        erelela_config_path=config.erelela_config,
        erelela_overrides=erelela_overrides,
        erelela_log_dir=os.path.join(config.log_dir, 'erelela'),
        erelela_enable_wandb=bool(config.use_wandb),
        erelela_wandb_project=config.project_name,
        use_wandb=bool(config.use_wandb),
        project_name=config.project_name,
    ))

    model = PPOTrainer(
        policy=PPOModel,
        env=venv,
        seed=config.run_id,
        run_id=config.run_id,
        can_see_walls=config.can_see_walls,
        image_noise_scale=config.image_noise_scale,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        model_n_epochs=config.model_n_epochs,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        gamma=config.gamma,
        discount=config.discount,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        batch_size=config.batch_size,
        pg_coef=config.pg_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        ext_rew_coef=config.ext_rew_coef,
        int_rew_source=config.int_rew_source,
        int_rew_coef=config.int_rew_coef,
        int_rew_decay=config.int_rew_decay,
        int_rew_norm=config.int_rew_norm,
        int_rew_momentum=config.int_rew_momentum,
        int_rew_eps=config.int_rew_eps,
        int_rew_clip=config.int_rew_clip,
        adv_momentum=config.adv_momentum,
        adv_norm=config.adv_norm,
        adv_eps=config.adv_eps,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        policy_kwargs=policy_kwargs,
        env_source=config.env_source,
        env_render=config.env_render,
        fixed_seed=config.fixed_seed,
        use_wandb=config.use_wandb,
        local_logger=config.local_logger,
        enable_plotting=config.enable_plotting,
        plot_interval=config.plot_interval,
        plot_colormap=config.plot_colormap,
        log_explored_states=config.log_explored_states,
        verbose=0,
    )

    if config.run_id == 0:
        print('model.policy:', model.policy)

    model.learn(
        total_timesteps=config.total_steps,
        callback=callbacks)


@click.command()
# Training params
@click.option('--run_id', default=0, type=int, help='Index (and seed) of the current run')
@click.option('--exp_name', default="test", type=str, help='The experiment name')
@click.option('--use_wandb', default=0, type=int, help='Whether using wandb')
@click.option('--log_dir', default='./logs', type=str, help='Directory for saving training logs')
@click.option('--total_steps', default=int(1e6), type=int, help='Total number of frames to run for training')
@click.option('--features_dim', default=64, type=int, help='Number of neurons of a learned embedding (PPO)')
@click.option('--model_features_dim', default=128, type=int,
              help='Number of neurons of a learned embedding (dynamics model)')
@click.option('--learning_rate', default=3e-4, type=float, help='Learning rate of PPO')
@click.option('--model_learning_rate', default=3e-4, type=float, help='Learning rate of the dynamics model')
@click.option('--num_processes', default=16, type=int, help='Number of training processes (workers)')
@click.option('--batch_size', default=512, type=int, help='Batch size')
@click.option('--n_steps', default=512, type=int, help='Number of steps to run for each process per update')
# Env params
@click.option('--env_source', default='minigrid', type=str, help='minigrid or procgen')
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, ninja, jumper')
@click.option('--project_name', required=False, type=str, help='Where to store training logs (wandb option)')
@click.option('--map_size', default=5, type=int, help='Size of the minigrid room')
@click.option('--can_see_walls', default=1, type=int, help='Whether walls are visible to the agent')
@click.option('--fully_obs', default=0, type=int, help='Whether the agent can receive full observations')
@click.option('--image_noise_scale', default=0.0, type=float, help='Standard deviation of the Gaussian noise')
@click.option('--procgen_mode', default='hard', type=str, help='Mode of ProcGen games (easy or hard)')
@click.option('--procgen_num_threads', default=4, type=int, help='Number of parallel ProcGen threads')
@click.option('--log_explored_states', default=0, type=int, help='Whether to log the number of explored states')
@click.option('--fixed_seed', default=-1, type=int, help='Whether to use a fixed env seed (MiniGrid)')
# Algo params
@click.option('--n_epochs', default=4, type=int, help='Number of epochs to train policy and value nets')
@click.option('--model_n_epochs', default=4, type=int, help='Number of epochs to train common_models')
@click.option('--gamma', default=0.99, type=float, help='Discount factor')
@click.option('--discount', default=0.99, type=float, help='Discount factor for discount sampling')
@click.option('--gae_lambda', default=0.95, type=float, help='GAE lambda')
@click.option('--pg_coef', default=1.0, type=float, help='Coefficient of policy gradients')
@click.option('--vf_coef', default=0.5, type=float, help='Coefficient of value function loss')
@click.option('--ent_coef', default=0.01, type=float, help='Coefficient of policy entropy')
@click.option('--max_grad_norm', default=0.5, type=float, help='Maximum norm of gradient')
@click.option('--clip_range', default=0.2, type=float, help='PPO clip range of the policy network')
@click.option('--clip_range_vf', default=-1, type=float,
              help='PPO clip range of the value function (-1: disabled, >0: enabled)')
@click.option('--adv_norm', default=2, type=int,
              help='Normalized advantages by: [0] No normalization [1] Standardization per mini-batch [2] Standardization per rollout buffer [3] Standardization w.o. subtracting the mean per rollout buffer')
@click.option('--adv_eps', default=1e-5, type=float, help='Epsilon for advantage normalization')
@click.option('--adv_momentum', default=0.9, type=float, help='EMA smoothing factor for advantage normalization')
# Reward params
@click.option('--ext_rew_coef', default=1.0, type=float, help='Coefficient of extrinsic rewards')
@click.option('--int_rew_coef', default=1e-2, type=float, help='Coefficient of intrinsic rewards (IRs)')
@click.option('--int_rew_decay', default='none', type=str, help='Decay coefficient of IRs by: none / linear / cos')
@click.option('--int_rew_source', default='DEIR', type=str,
              help='Source of IRs: [NoModel|TDD|Count|E3B|DEIR|ICM|RND|NGU|NovelD|PlainDiscriminator|PlainInverse|PlainForward]')
@click.option('--erelela_config', default=None, type=str, help='Path to the YAML config used to bootstrap EReLELA')
@click.option('--erelela_intrinsic_weight', default=0.1, type=float, help='Override ERELELA intrinsic weight if provided')
@click.option('--erelela_extrinsic_weight', default=20.0, type=float, help='Override ERELELA extrinsic weight if provided')
@click.option('--erelela_feedbacks_type', default='normal', type=str, help='Feedback type hint for EReLELA (matches config)')
@click.option('--erelela_override', multiple=True, default=None, type=str,
              help='Additional overrides for the EReLELA config formatted as KEY=VALUE (repeatable)')
@click.option('--use_baseline_ther_wrapper', default=0, type=int,
              help='Whether to wrap envs with baseline_ther_wrapper (0/1)')
@click.option('--baseline_ther_arg', multiple=True, default=None, type=str,
              help='Additional KEY=VALUE args forwarded to baseline_ther_wrapper (repeatable)')
@click.option('--int_rew_norm', default=1, type=int,
              help='Normalized IRs by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--int_rew_momentum', default=0.9, type=float,
              help='EMA smoothing factor for IR normalization (-1: total average)')
@click.option('--int_rew_eps', default=1e-5, type=float, help='Epsilon for IR normalization')
@click.option('--int_rew_clip', default=-1, type=float, help='Clip IRs into [-X, X] when X>0')
@click.option('--dsc_obs_queue_len', default=100000, type=int, help='Maximum length of observation queue (DEIR)')
@click.option('--icm_forward_loss_coef', default=0.2, type=float, help='Coefficient of forward model losses (ICM)')
@click.option('--ngu_knn_k', default=10, type=int, help='Search for K nearest neighbors (NGU)')
@click.option('--ngu_use_rnd', default=1, type=int, help='Whether to enable lifelong IRs generated by RND (NGU)')
@click.option('--ngu_dst_momentum', default=0.997, type=float,
              help='EMA smoothing factor for averaging embedding distances (NGU)')
@click.option('--rnd_use_policy_emb', default=1, type=int,
              help='Whether to use the embeddings learned by policy/value nets as inputs (RND)')
@click.option('--rnd_err_norm', default=1, type=int,
              help='Normalized RND errors by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--rnd_err_momentum', default=-1, type=float,
              help='EMA smoothing factor for RND error normalization (-1: total average)')
@click.option('--tdd_aggregate_fn', default='min', type=str, help='Aggregation function for TDD') # [min|avg|knn]
@click.option('--tdd_loss_fn', default="infonce_symmetric", type=str, help='loss function for TDD')
@click.option('--tdd_energy_fn', default='mrn_pot', type=str, help='energy function for TDD')
@click.option('--tdd_logsumexp_coef', default=0.0, type=float, help='logsumexp penalty for TDD')
@click.option('--offpolicy_data', default=0, type=int, help='whether to use offpolicy replay buffer')
# Network params
@click.option('--use_model_rnn', default=1, type=int, help='Whether to enable RNNs for the dynamics model')
@click.option('--latents_dim', default=256, type=int, help='Dimensions of latent features in policy/value nets\' MLPs')
@click.option('--model_latents_dim', default=256, type=int,
              help='Dimensions of latent features in the dynamics model\'s MLP')
@click.option('--policy_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--policy_mlp_layers', default=1, type=int, help='Number of latent layers used in the policy\'s MLP')
@click.option('--policy_cnn_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' CNN')
@click.option('--policy_mlp_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' MLP')
@click.option('--policy_gru_norm', default='NoNorm', type=str, help='Normalization type for policy/value nets\' GRU')
@click.option('--model_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--model_mlp_layers', default=1, type=int, help='Number of latent layers used in the model\'s MLP')
@click.option('--model_cnn_norm', default='BatchNorm', type=str,
              help='Normalization type for the dynamics model\'s CNN')
@click.option('--model_mlp_norm', default='BatchNorm', type=str,
              help='Normalization type for the dynamics model\'s MLP')
@click.option('--model_gru_norm', default='NoNorm', type=str, help='Normalization type for the dynamics model\'s GRU')
@click.option('--activation_fn', default='relu', type=str, help='Activation function for non-CNN layers')
@click.option('--cnn_activation_fn', default='relu', type=str, help='Activation function for CNN layers')
@click.option('--gru_layers', default=1, type=int, help='Number of GRU layers in both the policy and the model')
# Optimizer params
@click.option('--optimizer', default='adam', type=str, help='Optimizer, adam or rmsprop')
@click.option('--optim_eps', default=1e-5, type=float, help='Epsilon for optimizers')
@click.option('--adam_beta1', default=0.9, type=float, help='Adam optimizer option')
@click.option('--adam_beta2', default=0.999, type=float, help='Adam optimizer option')
@click.option('--rmsprop_alpha', default=0.99, type=float, help='RMSProp optimizer option')
@click.option('--rmsprop_momentum', default=0.0, type=float, help='RMSProp optimizer option')
# Logging & Analysis options
@click.option('--write_local_logs', default=1, type=int, help='Whether to output training logs locally')
@click.option('--enable_plotting', default=0, type=int, help='Whether to generate plots for analysis')
@click.option('--plot_interval', default=10, type=int, help='Interval of generating plots (iterations)')
@click.option('--plot_colormap', default='Blues', type=str, help='Colormap of plots to generate')
@click.option('--record_video', default=0, type=int, help='Whether to record video')
@click.option('--rec_interval', default=10, type=int, help='Interval of two videos (iterations)')
@click.option('--video_length', default=512, type=int, help='Length of the video (frames)')
@click.option('--log_dsc_verbose', default=0, type=int, help='Whether to record the discriminator loss for each action')
@click.option('--env_render', default=0, type=int, help='Whether to render games in human mode')
@click.option('--use_status_predictor', default=0, type=int,
    help='Whether to train status predictors for analysis (MiniGrid only)')
@click.option('--count_feedbacks_type', default='normal', 
    #choices=['normal', 'across-training', 'count-based', 'across-training-normal'],
    type=str, help='Type of count-based feedbacks')
@click.option('--force_gym_env', default=0, type=int,
    help='Force using Gym environment instead of custom environments')
@click.option('--use_legacy_env_wrapping', default=0, type=int,
    help='Disable StepAPI/Reset wrappers and rely on legacy env behavior (0/1)')
def main(
    run_id, exp_name, use_wandb, log_dir, total_steps, features_dim, model_features_dim, learning_rate, model_learning_rate,
    num_processes, batch_size, n_steps, env_source, game_name, project_name, map_size, can_see_walls, fully_obs,
    image_noise_scale, procgen_mode, procgen_num_threads, log_explored_states, fixed_seed, n_epochs, model_n_epochs,
    gamma, discount, gae_lambda, pg_coef, vf_coef, ent_coef, max_grad_norm, clip_range, clip_range_vf, adv_norm, adv_eps,
    adv_momentum, ext_rew_coef, int_rew_coef, int_rew_decay, int_rew_source, erelela_config, erelela_intrinsic_weight,
    erelela_extrinsic_weight, erelela_feedbacks_type, erelela_override, use_baseline_ther_wrapper, baseline_ther_arg,
    int_rew_norm, int_rew_momentum, int_rew_eps, int_rew_clip,
    dsc_obs_queue_len, icm_forward_loss_coef, ngu_knn_k, ngu_use_rnd, ngu_dst_momentum, rnd_use_policy_emb,
    rnd_err_norm, rnd_err_momentum, tdd_aggregate_fn, tdd_energy_fn, tdd_loss_fn, tdd_logsumexp_coef, offpolicy_data, use_model_rnn, latents_dim, model_latents_dim, policy_cnn_type, policy_mlp_layers,
    policy_cnn_norm, policy_mlp_norm, policy_gru_norm, model_cnn_type, model_mlp_layers, model_cnn_norm, model_mlp_norm,
    model_gru_norm, activation_fn, cnn_activation_fn, gru_layers, optimizer, optim_eps, adam_beta1, adam_beta2,
    rmsprop_alpha, rmsprop_momentum, write_local_logs, enable_plotting, plot_interval, plot_colormap, record_video,
    rec_interval, video_length, log_dsc_verbose, env_render, use_status_predictor, count_feedbacks_type,
    force_gym_env, use_legacy_env_wrapping,
):
    set_random_seed(run_id, using_cuda=True)
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)

    config.init_env_name(game_name, project_name)
    config.init_meta_info()
    config.init_logger()
    config.init_values()

    train(config)

    config.close()


if __name__ == '__main__':
    main()

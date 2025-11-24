import copy
import os
import time
from functools import partial
from typing import Dict, Optional

import torch as th
import wandb

import regym
from regym.environments import EnvType, generate_task
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from regym.rl_algorithms.algorithms.wrappers.ela_wrapper import ELAAlgorithmWrapper
from regym.rl_algorithms.networks import ArchiPredictorSpeaker, PreprocessFunction
from regym.rl_algorithms.agents.utils import generate_model, parse_and_check
from regym.util.wrappers import baseline_ther_wrapper

from impala_ride.algos.ela import load_configs


def _ensure_wandb_run(project: str, group: str, run_name: str, enable_wandb: bool) -> None:
    """Create a (possibly disabled) wandb run so regym logging does not crash."""
    if wandb.run is not None:
        return

    mode = "online" if enable_wandb else "disabled"
    wandb.init(project=project or "EReLELA", group=group, name=run_name, mode=mode)


def _extract(config: Dict, key: str, default=None):
    value = config
    for token in key.split('.'):
        if not isinstance(value, dict) or token not in value:
            return default
        value = value[token]
    return value


def _bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ["1", "true", "yes", "on"]
    return bool(value)


def _maybe_enable_augments(agent_config: Dict[str, any]) -> None:
    factor = float(agent_config.get("ELA_rg_logits_mdl_principle_factor", 0.0) or 0.0)
    betas = agent_config.get("ELA_rg_logits_mdl_principle_factor", None)
    if isinstance(betas, str) and "-" in betas:
        betas = [float(beta) for beta in betas.split('-')]
    else:
        betas = None

    if betas is not None or factor > 0.0:
        agent_config["ELA_rg_with_logits_mdl_principle"] = True
        if betas is not None:
            agent_config["ELA_rg_logits_mdl_principle_accuracy_threshold"] = 0.0

    if 'episodic-dissimilarity' in agent_config.get('ELA_rg_distractor_sampling', ''):
        agent_config['ELA_rg_same_episode_target'] = True

    if agent_config.get('ELA_rg_gaussian_blur_prob', 0.0) > 0.0:
        agent_config['ELA_rg_with_gaussian_blur_augmentation'] = True

    if agent_config.get('ELA_rg_color_jitter_prob', 0.0) > 0.0:
        agent_config['ELA_rg_with_color_jitter_augmentation'] = True

    if agent_config.get('ELA_rg_egocentric_prob', 0.0) > 0.0:
        agent_config['ELA_rg_egocentric'] = True

    language_specs = agent_config.get('ELA_rg_compactness_ambiguity_metric_language_specs', '')
    if 'natural' in language_specs:
        agent_config['THER_observe_achieved_goal'] = True

    if agent_config.get('language_guided_curiosity', False):
        agent_config['coverage_manipulation_metric'] = True
        if 'descr' not in agent_config.get('language_guided_curiosity_descr_type', ''):
            agent_config['MiniWorld_entity_visibility_oracle'] = True


def _build_pixel_wrapper(task_config: Dict[str, any], agent_config: Dict[str, any]):
    return partial(
        baseline_ther_wrapper,
        size=task_config.get('observation_resize_dim', 64),
        skip=task_config.get('nbr_frame_skipping', 1),
        stack=task_config.get('nbr_frame_stacking', 1),
        single_life_episode=task_config.get('single_life_episode', False),
        nbr_max_random_steps=task_config.get('nbr_max_random_steps', 0),
        clip_reward=task_config.get('clip_reward', False),
        time_limit=task_config.get('time_limit', None),
        max_sentence_length=agent_config.get('THER_max_sentence_length', 16),
        vocabulary=agent_config.get('THER_vocabulary', []),
        vocab_size=agent_config.get('THER_vocab_size', 64),
        previous_reward_action=task_config.get('previous_reward_action', False),
        observation_key=task_config.get('observation_key', 'image'),
        concatenate_keys_with_obs=task_config.get('concatenate_keys_with_obs', False),
        add_rgb_wrapper=task_config.get('add_rgb_wrapper', False),
        full_obs=task_config.get('full_obs', False),
        single_pick_episode=task_config.get('single_pick_episode', False),
        observe_achieved_pickup_goal=task_config.get('THER_observe_achieved_goal', False),
        use_visible_entities=False,
        babyai_mission=task_config.get('BabyAI_Bot_action_override', False),
        miniworld_symbolic_image=task_config.get('MiniWorld_symbolic_image', False),
        miniworld_entity_visibility_oracle=task_config.get('MiniWorld_entity_visibility_oracle', False),
        miniworld_entity_visibility_oracle_language_specs=task_config.get('MiniWorld_entity_visibility_oracle_language_specs', None),
        miniworld_entity_visibility_oracle_include_discrete_depth=task_config.get('MiniWorld_entity_visibility_oracle_include_discrete_depth', False),
        miniworld_entity_visibility_oracle_include_depth=task_config.get('MiniWorld_entity_visibility_oracle_include_depth', False),
        miniworld_entity_visibility_oracle_include_depth_precision=task_config.get('MiniWorld_entity_visibility_oracle_include_depth_precision', 0.1),
        miniworld_entity_visibility_oracle_too_far_threshold=task_config.get('MiniWorld_entity_visibility_oracle_too_far_threshold', 0.0),
        miniworld_entity_visibility_oracle_top_view=task_config.get('MiniWorld_entity_visibility_oracle_top_view', False),
        language_guided_curiosity=task_config.get('language_guided_curiosity', False),
        language_guided_curiosity_extrinsic_weight=task_config.get('language_guided_curiosity_extrinsic_weight', 1.0),
        language_guided_curiosity_intrinsic_weight=task_config.get('language_guided_curiosity_intrinsic_weight', 0.0),
        language_guided_curiosity_binary_reward=task_config.get('language_guided_curiosity_binary_reward', False),
        language_guided_curiosity_densify=task_config.get('language_guided_curiosity_densify', False),
        ne_count_based_exploration=task_config.get('language_guided_curiosity_non_episodic_count_based_exploration', False),
        ne_dampening_rate=task_config.get('language_guided_curiosity_non_episodic_dampening_rate', 1.0),
        coverage_manipulation_metric=task_config.get('coverage_manipulation_metric', False),
        descr_type=task_config.get('language_guided_curiosity_descr_type', ''),
    )


def build_erelela_wrapper(
    config_path: str,
    overrides: Optional[Dict[str, any]],
    run_dir: str,
    device: th.device,
    enable_wandb: bool,
    wandb_project: Optional[str],
    num_envs: int,
) -> ELAAlgorithmWrapper:
    overrides = overrides or {}
    experiment_config, agents_config, tasks_configs = load_configs(config_path, kwargs=overrides)
    assert len(tasks_configs) == 1, "EReLELA configs currently support a single task entry"

    task_config = copy.deepcopy(tasks_configs[0])
    agent_name = task_config['agent-id']
    env_name = task_config['env-id']
    run_name = task_config['run-id']
    agent_config = copy.deepcopy(agents_config[agent_name])

    # Apply CLI overrides directly on configs
    for key, value in (overrides or {}).items():
        agent_config[key] = value
        task_config[key] = value
        if key in task_config.get('env-config', {}):
            task_config['env-config'][key] = value

    agent_config['nbr_actor'] = num_envs
    task_config['nbr_actor'] = num_envs

    _maybe_enable_augments(agent_config)

    base_path = os.path.join(run_dir, env_name, run_name, agent_name)
    os.makedirs(base_path, exist_ok=True)
    experiment_config['experiment_id'] = base_path

    pixel_wrapping_fn = _build_pixel_wrapper(task_config, agent_config)
    regym.RegymSummaryWriterPath = base_path

    env_seed = experiment_config.get('env_seed', 0)
    task = generate_task(
        task_config['env-id'],
        env_type=EnvType.SINGLE_AGENT,
        nbr_parallel_env=task_config.get('nbr_actor', num_envs),
        wrapping_fn=pixel_wrapping_fn,
        test_wrapping_fn=pixel_wrapping_fn,
        env_config=task_config.get('env-config', {}),
        test_env_config=task_config.get('env-config', {}),
        seed=env_seed,
        test_seed=env_seed if task_config.get('static_envs', False) else env_seed + 100,
        static=task_config.get('static_envs', False),
        gathering=True,
    )

    agent_config['task_config'] = task_config
    agent_config['nbr_actor'] = task_config.get('nbr_actor', num_envs)

    kwargs = copy.deepcopy(agent_config)
    kwargs['discount'] = float(kwargs.get('discount', 0.99))
    kwargs['replay_capacity'] = int(float(kwargs.get('replay_capacity', 2048)))
    kwargs['min_capacity'] = int(float(kwargs.get('min_capacity', 64)))
    kwargs['use_cuda'] = _bool(kwargs.get('use_cuda', False))

    kwargs['state_preprocess'] = partial(PreprocessFunction, normalization=False)
    kwargs['goal_preprocess'] = partial(PreprocessFunction, normalization=False)

    kwargs = parse_and_check(kwargs, task)
    model = generate_model(task, kwargs)
    model = model.to(device=device)

    algorithm = R2D2Algorithm(
        kwargs=kwargs,
        model=model,
        name=f"{agent_name}_algo",
    )

    caption_predictor = ArchiPredictorSpeaker(
        model=model,
        **kwargs["ArchiModel"],
        pipeline_name="caption_generator",
        generator_name="CaptionGenerator",
    )
    caption_predictor = caption_predictor.to(device=device)

    group_name = f"erelela-{time.strftime('%Y%m%d-%H%M%S')}"
    _ensure_wandb_run(
        project=wandb_project or experiment_config.get('project', 'EReLELA'),
        group=group_name,
        run_name=f"erelela-rg-{run_name}",
        enable_wandb=enable_wandb,
    )

    ela_wrapper = ELAAlgorithmWrapper(
        algorithm=algorithm,
        predictor=caption_predictor,
        extrinsic_weight=kwargs['ELA_reward_extrinsic_weight'],
        intrinsic_weight=kwargs['ELA_reward_intrinsic_weight'],
        feedbacks={
            "failure": kwargs['ELA_feedbacks_failure_reward'],
            "success": kwargs['ELA_feedbacks_success_reward'],
        },
    )

    return ela_wrapper

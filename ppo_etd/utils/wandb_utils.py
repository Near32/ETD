import json
import logging
import os
import re
import socket
import uuid
from multiprocessing import current_process


def _load_wandb():
    try:
        import wandb  # type: ignore
    except ImportError:  # pragma: no cover - wandb is optional
        return None
    return wandb


def _infer_worker_suffix():
    process_name = current_process().name
    match = re.search(r"(\d+)$", process_name)
    if match:
        return match.group(1)
    return str(os.getpid())


def _build_child_run_name():
    context = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "host": socket.gethostname(),
        "process": current_process().name,
        "suffix": _infer_worker_suffix(),
        "token": uuid.uuid4().hex[:6],
    }
    template = os.environ.get("WANDB_CHILD_NAME_TEMPLATE")
    if template:
        try:
            return template.format(**context)
        except KeyError as exc:
            logging.warning("Invalid WANDB_CHILD_NAME_TEMPLATE placeholder %s", exc)
    prefix = os.environ.get("WANDB_CHILD_NAME_PREFIX", "env-worker")
    return f"{prefix}-{context['suffix']}"


def _start_child_run(wandb):
    project = os.environ.get("WANDB_CHILD_PROJECT") or os.environ.get("WANDB_PROJECT")
    if not project:
        logging.warning("Cannot start WandB child run: WANDB_CHILD_PROJECT/WANDB_PROJECT missing")
        return None

    entity = os.environ.get("WANDB_CHILD_ENTITY") or os.environ.get("WANDB_ENTITY")
    group = os.environ.get("WANDB_CHILD_GROUP")
    job_type = os.environ.get("WANDB_CHILD_JOB_TYPE", "env_worker")
    tags = [tag.strip() for tag in os.environ.get("WANDB_CHILD_TAGS", "").split(",") if tag.strip()]
    config = None
    config_json = os.environ.get("WANDB_CHILD_CONFIG_JSON")
    if config_json:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            logging.warning("Invalid WANDB_CHILD_CONFIG_JSON: %s", exc)

    settings_kwargs = {}
    start_method = os.environ.get("WANDB_CHILD_START_METHOD")
    if start_method:
        settings_kwargs["start_method"] = start_method
    mode = os.environ.get("WANDB_CHILD_MODE")
    if mode:
        settings_kwargs["mode"] = mode
    x_label = os.environ.get("WANDB_CHILD_X_LABEL")
    if x_label:
        settings_kwargs["x_label"] = x_label
    x_primary = os.environ.get("WANDB_CHILD_X_PRIMARY")
    if x_primary:
        settings_kwargs["x_primary"] = x_primary.lower() == "true"

    settings = wandb.Settings(**settings_kwargs) if settings_kwargs else None

    run_name = _build_child_run_name()
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            job_type=job_type,
            name=run_name,
            tags=tags or None,
            config=config,
            reinit=True,
            resume="never",
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - best-effort safeguard
        logging.warning("Failed to init WandB child run: %s", exc)
        return None
    return run


def _reattach_parent_run(wandb):
    run_id = os.environ.get("WANDB_RUN_ID")
    project = os.environ.get("WANDB_PROJECT")
    if not run_id or not project:
        return None

    entity = os.environ.get("WANDB_ENTITY")
    resume = os.environ.get("WANDB_RESUME", "allow")

    actor_id = int(os.environ.get("ACTOR_ID", 0))
    settings = wandb.Settings(
        x_label=f"learner_{actor_id}",
        mode="shared",
        x_primary=False,
    )
    try:
        run = wandb.init(
            id=run_id,
            project=project,
            entity=entity,
            resume=resume,
            reinit="return_previous",
            settings=settings,
        )
        os.environ["ACTOR_ID"] = str(actor_id + 1)
        return run
    except Exception as exc:  # pragma: no cover - best-effort safeguard
        logging.warning("Failed to reattach WandB run: %s", exc)
        return None


def ensure_wandb_initialized():
    wandb = _load_wandb()
    if wandb is None:
        return None

    if wandb.run is not None:
        return wandb.run

    child_mode = os.environ.get("WANDB_CHILD_RUN_MODE", "").lower()
    if child_mode in {"group", "child", "workers", "proc", "process"}:
        return _start_child_run(wandb)

    return _reattach_parent_run(wandb)

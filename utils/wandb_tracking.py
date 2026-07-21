"""Weights & Biases tracking for IRRA training.

Mirrors the structure of lab_clip's `src/wandb_tracking.py`: a thin
`WandbSession` wrapper that degrades to a no-op when logging is disabled, and
explicit `log_*` helpers that own the metric naming scheme.
"""
import json
import logging
import os
import os.path as op

try:
    import wandb
except ImportError:
    wandb = None

WANDB_META_FILENAME = "wandb_meta.json"
WANDB_RUN_ID_FILENAME = "wandb_run_id"

logger = logging.getLogger("IRRA.wandb")


class WandbSession():
    """No-op safe wrapper around a wandb run."""

    def __init__(self, run=None):
        self._run = run

    @property
    def enabled(self):
        return self._run is not None

    def log(self, metrics, step=None):
        if self._run is None:
            return
        self._run.log(metrics, step=step)

    def save(self, path, base_path=None):
        if self._run is None or not op.exists(path):
            return
        if base_path is None:
            self._run.save(path)
        else:
            self._run.save(path, base_path=base_path)

    def set_summary(self, metrics):
        if self._run is None:
            return
        for key, value in metrics.items():
            self._run.summary[key] = value

    def finish(self):
        if self._run is None:
            return
        self._run.finish()
        self._run = None


def parse_env_file(env_file):
    """Read a `KEY=value` env file into a dict. Missing file -> empty dict."""
    if not env_file or not op.exists(env_file):
        return {}

    values = {}
    with open(env_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            value = value.strip().strip('"').strip("'")
            if value:
                values[key.strip()] = value
    return values


def read_env_value(key, env_file):
    """Process environment wins over the env file."""
    value = os.environ.get(key)
    if value is not None:
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    return parse_env_file(env_file).get(key)


def _read_setting(args, key):
    value = getattr(args, key, "")
    if value is None:
        return ""
    return str(value).strip()


def is_wandb_enabled(args):
    return bool(getattr(args, "wandb", False))


def _scalar(value):
    """Meters hold raw loss tensors; detach before handing them to wandb."""
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    return float(value)


def flatten_config(config):
    """wandb config values must be scalars; JSON-encode everything else."""
    flat = {}
    for key, value in config.items():
        if key.startswith('_'):
            continue
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = value
    return flat


def _wandb_tags(args, *extra):
    raw_tags = getattr(args, "wandb_tags", None) or []
    tags = [str(tag) for tag in raw_tags]
    for tag in extra:
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def save_wandb_meta(output_dir, meta):
    os.makedirs(output_dir, exist_ok=True)
    with open(op.join(output_dir, WANDB_META_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    run_id = str(meta.get("run_id", "")).strip()
    if run_id:
        with open(op.join(output_dir, WANDB_RUN_ID_FILENAME), 'w', encoding='utf-8') as f:
            f.write(run_id)


def start_train_run(args):
    """Initialise the wandb run for a training job.

    Returns a disabled `WandbSession` when `--wandb` is not set, so callers can
    log unconditionally.
    """
    if not is_wandb_enabled(args):
        return WandbSession(None)
    if wandb is None:
        raise RuntimeError("wandb is not installed. Run: uv add wandb")

    env_file = _read_setting(args, "wandb_env_file") or "env/.env"

    api_key = read_env_value("WANDB_API_KEY", env_file)
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    else:
        logger.warning(
            "WANDB_API_KEY not found in environment or %s; "
            "relying on a pre-existing wandb login.", env_file)

    project = (_read_setting(args, "wandb_project")
               or read_env_value("WANDB_PROJECT", env_file) or "irra")
    entity = (_read_setting(args, "wandb_entity")
              or read_env_value("WANDB_ENTITY", env_file) or None)

    run_name = _read_setting(args, "wandb_run_name") or op.basename(args.output_dir)
    group = _read_setting(args, "wandb_group") or str(args.dataset_name)
    notes = _read_setting(args, "wandb_notes") or None
    tags = _wandb_tags(args, str(args.dataset_name), str(args.loss_names), "train")

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type="train",
        name=run_name,
        notes=notes,
        tags=tags,
        config=flatten_config(vars(args)),
        dir=args.output_dir,
    )
    session = WandbSession(run)
    save_wandb_meta(
        args.output_dir,
        {
            "run_id": run.id,
            "group": group,
            "project": run.project,
            "entity": run.entity or "",
            "job_type": "train",
            "output_dir": args.output_dir,
        })

    # `val/t2i_error@1` is the primary curve; make wandb rank runs by it.
    run.define_metric("epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/t2i_error@1", summary="min")
    run.define_metric("val/t2i_R1", summary="max")

    config_file = op.join(args.output_dir, 'configs.yaml')
    session.save(config_file, base_path=args.output_dir)
    return session


def log_train_epoch_metrics(session, epoch, meters, lr, temperature=None):
    """Per-epoch training averages (losses / accuracies) from the meters."""
    if not session.enabled:
        return
    payload = {"epoch": epoch, "train/lr": _scalar(lr)}
    for name, meter in meters.items():
        if meter.avg > 0:
            payload[f"train/{name}"] = _scalar(meter.avg)
    if temperature is not None:
        payload["train/temperature"] = _scalar(temperature)
    session.log(payload)


def log_val_metrics(session, epoch, metrics):
    """Per-epoch validation metrics, including the error curves.

    `metrics` uses the flat keys produced by `utils.metrics.Evaluator.eval`
    (e.g. `t2i_R1`, `t2i_mAP`, optionally the `i2t_*` counterparts).
    """
    if not session.enabled:
        return
    payload = {"epoch": epoch}
    for key, value in metrics.items():
        payload[f"val/{key}"] = _scalar(value)
    # Retrieval error = 100 - recall, so the curve goes down as training helps.
    for task in ("t2i", "i2t"):
        for rank in (1, 5, 10):
            key = f"{task}_R{rank}"
            if key in metrics:
                payload[f"val/{task}_error@{rank}"] = 100.0 - _scalar(metrics[key])
    session.log(payload)


def finish_train_run(session, best_top1, best_epoch, output_dir):
    if not session.enabled:
        return
    session.set_summary({
        "val/best_t2i_R1": _scalar(best_top1),
        "val/best_t2i_error@1": 100.0 - _scalar(best_top1),
        "val/best_epoch": best_epoch,
        "best_checkpoint": op.join(output_dir, 'best.pth'),
        "output_dir": output_dir,
    })
    session.finish()

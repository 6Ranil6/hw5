"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    low, high = (params["model"][k] for k in ("mps_low_watermark_ratio", "mps_high_watermark_ratio"))
    if not 0 < low <= high <= 1:
        raise ValueError("Нужны безопасные пороги MPS: 0 < low <= high <= 1")
    cfg = params["train"]
    for key in ("epochs", "batch_size", "grad_accum", "eval_every", "eval_batch_size"):
        if cfg[key] <= 0:
            raise ValueError(f"train.{key} должен быть положительным")
    if cfg["lr"] <= 0 or not 0 <= cfg["warmup_ratio"] <= 1:
        raise ValueError("Недопустимый lr или warmup_ratio")
    if params["lora"].get("modules_to_save"):
        raise ValueError("Для LoRA сохраняем только адаптер, modules_to_save запрещены")
    return params

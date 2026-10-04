"""Стадия train: LoRA-дообучение на выходе стадии tokenize.

    python -m src.train --variant all_layers          # полный прогон варианта
    python -m src.train --variant all_layers --max-steps 4 --out /tmp/x   # smoke

Цикл обучения написан руками, а не через Trainer: так видно всё, что
обычно прячется, — где считается val loss, как копятся градиенты, что
сохраняется рядом с адаптером.

Что сохраняется в models/adapter_<variant>/: адаптер.
В metrics/train_<variant>.json — кривые train/val loss, время, пиковая память,
число обучаемых параметров, вес адаптера и отпечаток входов.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Ограничения Metal задаются конфигом до первого импорта torch.
from src.config import load_params

_mps_config = load_params()["model"]
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1" if _mps_config["mps_prefer_metal"] else "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(_mps_config["mps_high_watermark_ratio"]))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(_mps_config["mps_low_watermark_ratio"]))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.checkpoint import save_checkpoint, restore_checkpoint
from src.data import LABEL_PAD_ID, batches, load_split, accumulation_groups
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed, synchronize


TRAIN_CODE = ("src/checkpoint.py", "src/train.py", "src/data.py", "src/runtime.py", "src/config.py", "uv.lock", "pyproject.toml")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants", "compare")


def inputs_fingerprint(params: dict) -> str:
    """Отпечаток кода обучения и секций конфига, от которых зависит адаптер.

    check.sh сверяет его с текущим: правили train.py или lr, а адаптер
    остался от прошлого прогона — проверять его бессмысленно.
    """
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    for split in ("train", "val"):
        h.update(Path(params["data"][split]).read_bytes())
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    cfg = params["lora"]
    if not 0 <= freeze_first < n_layers:
        raise ValueError("freeze_first должен оставлять хотя бы один обучаемый слой")
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=None,
        layers_to_transform=list(range(freeze_first, n_layers)),
        revision=params["model"].get("revision"),
        task_type="CAUSAL_LM",
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int, observe_memory=None, pad_multiple=1) -> float:
    """Средний лосс на токен по всему val-сплиту.

    Среднее по батчам нельзя: в батчах разное число токенов под маской.
    Поэтому сумма лоссов, взвешенная числом токенов, делённая на их сумму.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0, pad_multiple=pad_multiple):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
        if n == 0:
            continue
        loss = model(**batch).loss
        if observe_memory:
            observe_memory()
        total += loss.item() * n
        count += n
    model.train(was_training)
    if device.type == "mps":
        torch.mps.empty_cache()   # логиты оценки не должны висеть в кэше до конца обучения
    if not count:
        raise ValueError("В validation нет токенов ответа под лоссом")
    return total / count


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="куда писать адаптер и метрики (smoke-тесты)")
    ap.add_argument("--val-limit", type=int, default=None, help="оценивать на первых N примерах val (smoke)")
    args = ap.parse_args()

    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}, есть: {sorted(variants)}")
    variant = variants[args.variant]
    tcfg = params["train"]
    set_seed(tcfg["seed"])
    run_fingerprint = inputs_fingerprint(params)
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")

    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])

    train_blob = load_split(params["data"]["train"])
    val_blob = load_split(params["data"]["val"])
    if args.val_limit is not None and args.val_limit <= 0:
        raise ValueError("val-limit должен быть положительным")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max-steps должен быть положительным")
    if args.val_limit:
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
    pad_id = train_blob["pad_token_id"]

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"], revision=params["model"].get("revision"))
    model = AutoModelForCausalLM.from_pretrained(params["model"]["name"], dtype=dtype, revision=params["model"].get("revision"), attn_implementation=tcfg["attn_implementation"]).to(device)
    tokenizer.padding_side = train_blob["padding_side"]
    if train_blob["model"] != params["model"]["name"] or val_blob["model"] != params["model"]["name"]:
        raise ValueError("Тензоры подготовлены для другой модели")
    if tokenizer.pad_token_id != pad_id or val_blob["pad_token_id"] != pad_id:
        raise ValueError("pad_token_id входа не совпадает с токенизатором")
    n_layers = model.config.num_hidden_layers
    if params["train"].get("gradient_checkpointing"):
        # Активации 28 слоёв не храним, а пересчитываем на обратном проходе:
        # памяти в разы меньше, шаг примерно на треть дольше.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, n_layers, variant["freeze_first"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    examples = train_blob["examples"]
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps
    )

    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    peak = allocated_bytes(device)

    def observe_memory():
        nonlocal peak
        peak = max(peak, allocated_bytes(device))

    checkpoint_path = Path("checkpoints") / f"{args.variant}.pt"
    recoverable = args.out is None and args.max_steps is None and args.val_limit is None
    expected = dict(fingerprint=run_fingerprint, variant=args.variant, device=str(device), total_steps=total_steps)
    previous_wall = 0.0
    step, tokens, supervised_tokens, seen_examples, diverged = 0, 0, 0, 0, False
    if recoverable and checkpoint_path.exists():
        state = restore_checkpoint(checkpoint_path, model, optimizer, scheduler, expected, device)
        step, tokens, supervised_tokens, seen_examples = (state[k] for k in ("step", "tokens", "supervised_tokens", "seen_examples"))
        curve_train, curve_val = state["curve_train"], state["curve_val"]
        base_val, eval_seconds, previous_wall = state["base_val"], state["eval_seconds"], state["wall_seconds"]
        peak = max(peak, state["peak"])
        print(f"[{args.variant}] restored checkpoint at step {step}/{total_steps}", flush=True)
    else:
        synchronize(device)
        eval_start = time.perf_counter()
        base_val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs, observe_memory, tcfg["pad_multiple"])
        synchronize(device)
        eval_seconds = time.perf_counter() - eval_start
        previous_wall = eval_seconds
        curve_train = []
        curve_val = [[0, round(base_val, 6)]]
    synchronize(device)
    started = time.perf_counter()
    resumed_step = step
    print(f"[{args.variant}] устройство {device}, обучаемых {trainable:,} из {total:,} "
          f"({trainable / total:.3%}); шагов {total_steps}; base val {base_val:.6f}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(tcfg["epochs"]):
        stream = batches(examples, tcfg["batch_size"], pad_id, shuffle=True, seed=tcfg["seed"] + epoch, pad_multiple=tcfg["pad_multiple"])
        for group_index, group in enumerate(accumulation_groups(stream, tcfg["grad_accum"])):
            if epoch * steps_per_epoch + group_index < resumed_step:
                continue
            # Средний лосс по токенам ответа всего эффективного батча.
            # Последняя неполная группа тоже делает шаг и имеет верный вес.
            counts = [int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum()) for batch in group]
            n_supervised = sum(counts)
            if not n_supervised:
                raise ValueError("Эффективный батч не содержит токенов ответа")
            accum_loss = 0.0
            for batch, count in zip(group, counts):
                tokens += int(batch["attention_mask"].sum())
                seen_examples += batch["input_ids"].shape[0]
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Нечисловой loss на шаге {step + 1}")
                (loss * (count / n_supervised)).backward()
                accum_loss += value * count / n_supervised
                observe_memory()
            supervised_tokens += n_supervised
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"], error_if_nonfinite=True)
            except RuntimeError as error:
                bad = [name for name, param in model.named_parameters()
                       if param.grad is not None and not torch.isfinite(param.grad.detach().cpu()).all()]
                raise RuntimeError(f"Шаг {step + 1}: нечисловые градиенты в {bad[:10]}") from error
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss, 4)])
            print(f"  train шаг {step}/{total_steps}: loss {accum_loss:.4f}, grad_norm {float(grad_norm):.4f}", flush=True)
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                synchronize(device)
                eval_start = time.perf_counter()
                val_loss = evaluate(model, val_blob["examples"], pad_id, device, eval_bs, observe_memory, tcfg["pad_multiple"])
                synchronize(device)
                eval_seconds += time.perf_counter() - eval_start
                curve_val.append([step, round(val_loss, 6)])
                print(f"  шаг {step}/{total_steps}: train {accum_loss:.4f}, val {val_loss:.6f}", flush=True)
            if recoverable and (step % tcfg["eval_every"] == 0 or step == total_steps):
                synchronize(device)
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, dict(
                    **expected, step=step, tokens=tokens, supervised_tokens=supervised_tokens,
                    seen_examples=seen_examples, curve_train=curve_train, curve_val=curve_val,
                    base_val=base_val, eval_seconds=eval_seconds, peak=peak,
                    wall_seconds=previous_wall + time.perf_counter() - started,
                ), device)
                print(f"  checkpoint saved: step {step}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    synchronize(device)
    wall_seconds = previous_wall + time.perf_counter() - started
    seconds = wall_seconds - eval_seconds

    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    if run_fingerprint != inputs_fingerprint(load_params()):
        raise RuntimeError("Код, конфигурация или данные изменились во время обучения")
    model.save_pretrained(adapter_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "inference_config.json").write_text(json.dumps({
        "model": params["model"], "compare": params["compare"],
        "inputs_fingerprint": run_fingerprint,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "variant": args.variant,
        "freeze_first": variant["freeze_first"],
        "model": params["model"]["name"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "base_val_loss": base_val,
        "final_val_loss": curve_val[-1][1] if curve_val else None,
        "diverged": diverged,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "seconds": round(seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "seconds_per_step": round(seconds / max(step, 1), 3),
        "train_tokens_per_sec": round(tokens / seconds, 1) if seconds else 0,
        "processed_tokens": tokens,
        "processed_supervised_tokens": supervised_tokens,
        "processed_examples": seen_examples,
        "val_examples": len(val_blob["examples"]),
        "max_steps_override": args.max_steps,
        "val_limit_override": args.val_limit,
        "wall_seconds": round(wall_seconds, 1),
        "layers_to_transform": list(range(variant["freeze_first"], n_layers)),
        "peak_memory_mb": round(peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": run_fingerprint,
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {step} шагов за {seconds:.0f} с; "
          f"пик памяти {metrics['peak_memory_mb']:.0f} МБ; адаптер {metrics['adapter_size_mb']} МБ -> {adapter_dir}")


if __name__ == "__main__":
    main()

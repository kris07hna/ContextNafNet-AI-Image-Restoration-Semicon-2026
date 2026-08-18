from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import (
    ModelEMA,
    load_checkpoint,
    restore_random_state,
    save_checkpoint,
    widen_input_weights,
)
from .distributed import DistributedInfo, reduce_sum
from .losses import PIXEL_MODES, RestorationLoss
from .metrics import psnr, ssim
from .models import ModelConfig

INPUT_FIELDS = frozenset({"input_mode", "in_channels", "noise_quadratic", "noise_linear",
                          "noise_constant", "vst_margin", "noise_blur_sigma"})

LossState = tuple[tuple[float, float, float, float], str]


def loss_schedule(settings: dict, criterion: RestorationLoss) -> Callable[[float], LossState]:
    # The loss changes shape twice over a run, so both transitions are resolved here rather than in the
    # training loop: the frequency term ramps in only once the model has learned enough structure for a
    # spectral penalty to be informative, and an optional final phase switches the pixel term to
    # squared error. PSNR is a monotone function of the mean squared error, so that tail converts
    # structure the perceptual schedule already found into reported metric.
    base_weights, base_pixel = tuple(float(value) for value in criterion.weights[:3]), criterion.pixel_mode
    target = float(criterion.weights[3])
    start = float(settings.get("frequency_start_fraction", 0.0))
    ramp = float(settings.get("frequency_ramp_fraction", 0.0))
    configured = settings.get("final_phase_weights")
    final_fraction = float(settings.get("final_phase_fraction", 0.9))
    final_pixel = str(settings.get("final_phase_pixel_loss", "mse"))
    final: LossState | None = None
    if configured is not None:
        if len(configured) != 4:
            raise ValueError(f"final_phase_weights needs four values, got {len(configured)}")
        if final_pixel not in PIXEL_MODES:
            raise ValueError(f"Unknown final_phase_pixel_loss: {final_pixel}")
        if not 0.0 < final_fraction <= 1.0:
            raise ValueError(f"final_phase_fraction must be in (0, 1], got {final_fraction}")
        final = (tuple(float(value) for value in configured), final_pixel)

    def plan(progress: float) -> LossState:
        if final is not None and progress >= final_fraction:
            return final
        if progress < start:
            frequency = 0.0
        elif ramp > 0:
            frequency = target * min(1.0, (progress - start) / ramp)
        else:
            frequency = target
        return (*base_weights, frequency), base_pixel

    return plan


def autocast_context(device: torch.device, enabled: bool, precision: str):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool, precision: str,
             distributed: DistributedInfo | None = None) -> dict[str, float]:
    distributed = distributed or DistributedInfo()
    was_training = model.training
    model.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "count": 0.0}
    validation_started = time.perf_counter()
    if distributed.primary:
        print(f"validation_start images={len(loader.dataset)} batches={len(loader)}", flush=True)
    for batch_index, batch in enumerate(loader, start=1):
        lr, gt = batch["lr"].to(device, non_blocking=True), batch["gt"].to(device, non_blocking=True)
        with autocast_context(device, amp, precision):
            prediction = model(lr)
        totals["psnr"] += float(psnr(prediction.float(), gt).sum())
        totals["ssim"] += float(ssim(prediction.float(), gt).sum())
        totals["count"] += float(lr.shape[0])
        if distributed.primary and (batch_index % 50 == 0 or batch_index == len(loader)):
            print(
                f"validation_progress batch={batch_index}/{len(loader)} "
                f"elapsed={time.perf_counter() - validation_started:.1f}s",
                flush=True,
            )
    model.train(was_training)
    # Summing totals and counts before dividing keeps the reduced metric identical to the value a
    # single process would report, which averaging per-rank means would only approximate.
    totals = reduce_sum(totals, distributed)
    return {"psnr": totals["psnr"] / totals["count"], "ssim": totals["ssim"] / totals["count"]}


def train(model: torch.nn.Module, train_loader: DataLoader, validation_loader: DataLoader, config: dict,
          model_config: dict, device: torch.device, precision: str, resume: str | None = None,
          initialize: str | None = None, distributed: DistributedInfo | None = None) -> dict:
    distributed = distributed or DistributedInfo()
    primary = distributed.primary
    settings = config["training"]
    def canonical_model_config(values: dict) -> dict:
        return ModelConfig.from_dict(values).to_dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    updates_per_epoch = math.ceil(len(train_loader) / settings["gradient_accumulation"])
    total_steps = max(1, updates_per_epoch * settings["epochs"])
    warmup = max(0, int(settings.get("warmup_steps", 0)))
    minimum_lr_ratio = float(settings.get("minimum_lr", 0.0)) / float(settings["lr"])
    def lr_factor(step: int) -> float:
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1 / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_lr_ratio + (1 - minimum_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == "fp16")
    criterion = RestorationLoss(
        pixel_weight=float(settings.get("pixel_loss_weight", 0.70)),
        ssim_weight=float(settings.get("ssim_loss_weight", 0.20)),
        gradient_weight=float(settings.get("gradient_loss_weight", 0.10)),
        frequency_weight=float(settings.get("frequency_loss_weight", 0.0)),
        structural_mode=str(settings.get("structural_loss", "ssim")),
        edge_weight=float(settings.get("edge_loss_emphasis", 0.0)),
        frequency_mode=str(settings.get("frequency_loss", "haar")),
        pixel_mode=str(settings.get("pixel_loss", "charbonnier")),
        lpips_weight=float(settings.get("lpips_loss_weight", 0.0)),
        lpips_net=str(settings.get("lpips_net", "vgg")),
    )
    schedule = loss_schedule(settings, criterion)
    ema = ModelEMA(model, settings["ema_decay"])
    output_dir, report_dir = Path(config["output_dir"]), Path(config["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    best = {"psnr": float("-inf"), "ssim": float("-inf")}
    step, start_epoch, started = 0, 0, time.perf_counter()
    if resume and initialize:
        raise ValueError("Use either resume or initialize, not both")
    if initialize:
        checkpoint = load_checkpoint(initialize, "cpu")
        saved, wanted = canonical_model_config(checkpoint["model_config"]), canonical_model_config(model_config)
        # Initialization only seeds weights, so the input representation is allowed to differ; every
        # field that changes the body of the network must still match exactly.
        body = {key: value for key, value in saved.items() if key not in INPUT_FIELDS}
        if body != {key: value for key, value in wanted.items() if key not in INPUT_FIELDS}:
            differing = sorted(key for key in wanted if key not in INPUT_FIELDS and saved.get(key) != wanted[key])
            raise ValueError(f"Initialization checkpoint model configuration does not match: {differing}")
        state, notes = widen_input_weights(checkpoint["ema"], model)
        model.load_state_dict(state, strict=True)
        ema.module.load_state_dict(state, strict=True)
        if primary:
            print(f"Initialized model and EMA from {initialize}"
                  + (f" (adapted {', '.join(notes)})" if notes else ""))
    if resume:
        checkpoint = load_checkpoint(resume, "cpu")
        if canonical_model_config(checkpoint["model_config"]) != canonical_model_config(model_config):
            raise ValueError("Resume checkpoint model configuration does not match")
        model.load_state_dict(checkpoint["model"])
        ema.module.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step, start_epoch, best = checkpoint["step"], checkpoint["epoch"], checkpoint["best"]
        if step >= (start_epoch + 1) * updates_per_epoch:
            start_epoch += 1
        restore_random_state(checkpoint["random_state"])
        if primary:
            print(f"Resumed {resume} at epoch={start_epoch}, step={step}")
    # Wrapping happens after the weights are loaded and before the first forward pass. The unwrapped
    # module stays the reference for the EMA and for checkpointing, so a two-GPU run writes exactly the
    # same checkpoint layout as a single-GPU run and either can load the other's output.
    core = model
    if distributed.enabled:
        model = DistributedDataParallel(model, device_ids=[distributed.device] if distributed.cuda else None)
        if primary:
            print(f"distributed backend=ddp world_size={distributed.world_size}")
    optimizer.zero_grad(set_to_none=True)
    log_path = report_dir / "training.jsonl"
    curriculum = settings.get("crop_curriculum", [])
    def crop_size_for_epoch(epoch: int) -> int | None:
        if not curriculum:
            return settings.get("lr_crop_size")
        progress = epoch / max(1, settings["epochs"] - 1)
        for phase in curriculum:
            if progress <= float(phase["until"]):
                sizes = [int(size) for size in phase["sizes"]]
                weights = [float(weight) for weight in phase.get("weights", [1] * len(sizes))]
                rng = torch.Generator().manual_seed(int(config["seed"]) + epoch)
                return sizes[int(torch.multinomial(torch.tensor(weights), 1, generator=rng).item())]
        return int(curriculum[-1]["sizes"][-1])
    for epoch in range(start_epoch, settings["epochs"]):
        if step >= total_steps:
            break
        crop_size = crop_size_for_epoch(epoch)
        if hasattr(train_loader.dataset, "set_crop_size"):
            train_loader.dataset.set_crop_size(crop_size)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        if primary:
            print(
                f"epoch_start epoch={epoch + 1}/{settings['epochs']} step={step}/{total_steps} "
                f"batches={len(train_loader)} lr_crop_size={crop_size}",
                flush=True,
            )
        for batch_index, batch in enumerate(train_loader):
            if step >= total_steps:
                break
            lr, gt = batch["lr"].to(device, non_blocking=True), batch["gt"].to(device, non_blocking=True)
            criterion.weights, criterion.pixel_mode = schedule(step / max(1, total_steps))
            should_update = (batch_index + 1) % settings["gradient_accumulation"] == 0 or batch_index + 1 == len(train_loader)
            # Accumulation steps that are not about to update do not need their gradients reduced, so
            # suppressing the all-reduce until the last micro-batch cuts the collective traffic by the
            # accumulation factor without changing the gradient that reaches the optimizer.
            reduction = nullcontext() if should_update or not distributed.enabled else model.no_sync()
            with reduction:
                with autocast_context(device, amp, precision):
                    prediction = model(lr)
                    loss, parts = criterion(prediction, gt)
                    loss = loss / settings["gradient_accumulation"]
                if not torch.isfinite(loss):
                    values = {key: float(value) for key, value in parts.items()}
                    raise FloatingPointError(f"Non-finite loss at step={step}, epoch={epoch + 1}: {values}")
                scaler.scale(loss).backward()
            if not should_update:
                continue
            if settings.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(core)
            step += 1
            record = {"epoch": epoch, "step": step, "lr_crop_size": crop_size,
                      "pixel_mode": criterion.pixel_mode, "pixel_weight": criterion.weights[0],
                      "frequency_weight": criterion.weights[3],
                      "synthetic_fraction": float(batch["synthetic"].mean()) if "synthetic" in batch else 0.0,
                      "loss": float(loss.detach()) * settings["gradient_accumulation"],
                      "lr": scheduler.get_last_lr()[0], **{key: float(value) for key, value in parts.items()}}
            if primary:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            log_every = max(1, int(settings.get("log_every_steps", 25)))
            if primary and (step == 1 or step % log_every == 0):
                print(
                    f"train epoch={epoch + 1}/{settings['epochs']} step={step}/{total_steps} "
                    f"loss={record['loss']:.5f} lr={record['lr']:.3e}",
                    flush=True,
                )
            validate_now = step % settings["validate_every_steps"] == 0 or step == total_steps
            if validate_now:
                # Every rank runs the reduced validation, so every rank derives the same metrics and the
                # same best-so-far record; only rank zero writes, which keeps the files uncontended.
                metrics = validate(ema.module, validation_loader, device, amp, precision, distributed)
                if primary:
                    print(
                        f"validation_result step={step} loss={record['loss']:.5f} "
                        f"psnr={metrics['psnr']:.3f} ssim={metrics['ssim']:.5f}",
                        flush=True,
                    )
                improved = metrics["psnr"] > best["psnr"]
                best = {key: max(best[key], metrics[key]) for key in best}
                if primary:
                    save_checkpoint(output_dir / "latest.pt", core, ema, optimizer, scheduler, scaler,
                                    model_config, config, step, epoch, best)
                    if improved:
                        save_checkpoint(output_dir / "best.pt", core, ema, optimizer, scheduler, scaler,
                                        model_config, config, step, epoch, best)
            elif primary and step % settings["save_every_steps"] == 0:
                save_checkpoint(output_dir / "latest.pt", core, ema, optimizer, scheduler, scaler,
                                model_config, config, step, epoch, best)
    return {"steps": step, "elapsed_seconds": time.perf_counter() - started, **best}

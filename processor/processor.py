import logging
import os
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, get_world_size, synchronize
from utils.ema import ModelEMA
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from utils.wandb_tracking import (
    WandbSession,
    log_train_epoch_metrics,
    log_val_metrics,
)
from utils.efficiency import (
    build_epoch_efficiency_metrics,
    finish_cuda_timer,
    format_peak_vram,
    get_global_processed_examples,
    get_peak_vram_metrics,
    start_measurement,
)


def _evaluate_with_efficiency(evaluator, model, device):
    started_at = start_measurement(device)
    metrics = evaluator.eval(
        model.eval(),
        i2t_metric=True,
        return_metrics=True,
    )
    epoch_seconds = finish_cuda_timer(device, started_at)
    vram_metrics = get_peak_vram_metrics(device)
    return metrics, {"epoch_seconds": epoch_seconds}, vram_metrics


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, wandb_session=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = torch.device("cuda")
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    if wandb_session is None:
        wandb_session = WandbSession(None)

    amp_enabled = getattr(args, 'amp', False)
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    ema = None
    if getattr(args, 'ema', False) and get_rank() == 0:
        ema_source = model.module if args.distributed else model
        ema = ModelEMA(ema_source, decay=getattr(args, 'ema_decay', 0.999))
        logger.info(f"EMA enabled, decay: {ema.decay}")

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    cumulative_gpu_seconds = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        for meter in meters.values():
            meter.reset()
        model.train()
        train_started_at = start_measurement(device)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                ret = model(batch)
                total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.module if args.distributed else model)
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        train_seconds = finish_cuda_timer(device, train_started_at)
        train_vram_metrics = get_peak_vram_metrics(device)
        processed_examples = get_global_processed_examples(
            meters["loss"].count,
            device,
        )
        cumulative_gpu_seconds += train_seconds * get_world_size()
        train_efficiency_metrics = build_epoch_efficiency_metrics(
            epoch_seconds=train_seconds,
            processed_examples=processed_examples,
            cumulative_seconds=cumulative_gpu_seconds,
        )

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        if get_rank() == 0:
            log_train_epoch_metrics(wandb_session,
                                    epoch=epoch,
                                    meters=meters,
                                    lr=scheduler.get_lr()[0],
                                    temperature=ret['temperature'],
                                    efficiency_metrics=train_efficiency_metrics,
                                    vram_metrics=train_vram_metrics)


        scheduler.step()
        if get_rank() == 0:
            time_per_batch = train_seconds / (n_iter + 1)
            logger.info(
                "Epoch {} done. Train time: {:.3f}[s] Time per batch: {:.3f}[s] "
                "Speed: {:.1f}[samples/s]{}"
                .format(
                    epoch,
                    train_seconds,
                    time_per_batch,
                    train_efficiency_metrics["examples_per_second"],
                    format_peak_vram(train_vram_metrics),
                ))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                eval_model = ema.module if ema is not None else (model.module if args.distributed else model)
                val_metrics, val_efficiency_metrics, val_vram_metrics = (
                    _evaluate_with_efficiency(
                        evaluator=evaluator,
                        model=eval_model,
                        device=device,
                    )
                )
                top1 = val_metrics['t2i_R1']

                for k, v in val_metrics.items():
                    tb_writer.add_scalar(f'val/{k}', v, epoch)
                log_val_metrics(
                    wandb_session,
                    epoch=epoch,
                    metrics=val_metrics,
                    efficiency_metrics=val_efficiency_metrics,
                    vram_metrics=val_vram_metrics,
                )
                logger.info(
                    "Validation epoch {} done. Time: {:.3f}[s]{}"
                    .format(
                        epoch,
                        val_efficiency_metrics["epoch_seconds"],
                        format_peak_vram(val_vram_metrics),
                    ))

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                    if ema is not None:
                        ema_save_path = os.path.join(args.output_dir, "best_ema.pth")
                        torch.save({"model": ema.state_dict(), "epoch": epoch}, ema_save_path)
                        logger.info(f"Saving EMA checkpoint to {ema_save_path}")
    if get_rank() == 0:
        best_epoch = arguments.get("epoch", start_epoch)
        logger.info(f"best R1: {best_top1} at epoch {best_epoch}")
    return best_top1, arguments.get("epoch", start_epoch)


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())

import os
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchaudio
import random
import commons
import periodwave.inference_utils.utils as utils
from dataset_codec import MelDataset, mel_spectrogram, MAX_WAV_VALUE
from torch.utils.data.distributed import DistributedSampler
import auraloss
from pesq import pesq

from periodwave.periodwave_encodec_turbo import FlowMatch
from periodwave.bigvganv2_discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from periodwave.ms_stftd import MultiScaleSTFTDiscriminator

import itertools
from periodwave.inference_utils.encodec_feature_extractor import EncodecFeatures

torch.backends.cudnn.benchmark = True
global_step = 0


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0, 100)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    hps = utils.get_hparams()
    if n_gpus > 1:
        mp.spawn(
            run,
            nprocs=n_gpus,
            args=(
                n_gpus,
                hps,
            ),
        )
    else:
        run(0, n_gpus, hps)


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
    if n_gpus > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
        )

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{:d}".format(rank))

    train_dataset = MelDataset(
        hps.data.train_filelist_path,
        hps,
        hps.train.segment_size,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
        n_cache_reuse=0,
        shuffle=False if n_gpus > 1 else True,
        device=device,
    )

    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=4,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    if rank == 0:
        test_dataset = MelDataset(
            hps.data.test_filelist_path,
            hps,
            hps.train.segment_size,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            n_cache_reuse=0,
            split=False,
            shuffle=False,
            device=device,
        )
        eval_loader = DataLoader(
            test_dataset,
            num_workers=1,
            batch_size=1,
            pin_memory=True,
            persistent_workers=True,
        )

    Encodec = EncodecFeatures().cuda()

    model = FlowMatch(
        hps.data.n_mel_channels,
        hps.model.periods,
        hps.model.noise_scale,
        hps.model.final_dim,
        hps.model.hidden_dim,
    ).cuda()

    # define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(hps.discriminator).cuda()
    mrd = MultiScaleSubbandCQTDiscriminator(hps.discriminator).cuda()
    ms_stftd = (
        MultiScaleSTFTDiscriminator().cuda()
    )  # We additionally utilized MS-STFTD for EnCodec version.

    if rank == 0:
        num_param = get_param_num(model)
        print("number of Parameters:", num_param)

    optimizer = torch.optim.AdamW(model.parameters(), hps.train.learning_rate)

    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters(), ms_stftd.parameters()),
        hps.train.learning_rate,
    )

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), model, optimizer
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "MPD_*.pth"), mpd
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "MRD_*.pth"), mrd
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "MSTFTD_*.pth"),
            ms_stftd,
            optim_d,
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        _, _, _, epoch_str = utils.load_checkpoint(hps.train.pretrain_path, model)

        epoch_str = 1
        global_step = 0

    if n_gpus > 1:
        model = DDP(model, device_ids=[rank])
        mpd = DDP(mpd, device_ids=[rank]).to(device)
        mrd = DDP(mrd, device_ids=[rank]).to(device)
        ms_stftd = DDP(ms_stftd, device_ids=[rank]).to(device)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                model,
                [mpd, mrd, ms_stftd],
                [optimizer, optim_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                writer,
                n_gpus,
                Encodec,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                model,
                [mpd, mrd, ms_stftd],
                [optimizer, optim_d],
                scaler,
                [train_loader, None],
                None,
                None,
                n_gpus,
                Encodec,
            )


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    discs,
    optims,
    scaler,
    loaders,
    logger,
    writers,
    n_gpus,
    Encodec,
):
    model = nets
    mpd, mrd, ms_stftd = discs
    optimizer, optim_d = optims
    train_loader, eval_loader = loaders

    if writers is not None:
        writer = writers

    global global_step

    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)

    model.train()
    mpd.train()
    mrd.train()
    ms_stftd.train()

    for batch_idx, (y, length) in enumerate(train_loader):
        y = y.cuda(rank, non_blocking=True)
        length = length.cuda(rank, non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad():
            embs = Encodec(y)  # [B, K = 8, T]

        if n_gpus > 1:
            loss_msmel, fake_y = model.module.compute_loss(
                y,
                embs,
                length,
                hps.train.tuning_steps,
                hps.train.finetuning_temperature,
            )

        else:
            loss_msmel, fake_y = model.compute_loss(
                y,
                embs,
                length,
                hps.train.tuning_steps,
                hps.train.finetuning_temperature,
            )

        optim_d.zero_grad()
        y = y.unsqueeze(1)
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, fake_y.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )

        # MRD
        y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, fake_y.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        # MS_STFTD
        y_ds_hat_r2, y_ds_hat_g2, _, _ = ms_stftd(y, fake_y.detach())
        loss_disc_s2, losses_disc_s_r2, losses_disc_s_g2 = discriminator_loss(
            y_ds_hat_r2, y_ds_hat_g2
        )

        loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_s2
        loss_disc_all.backward()
        grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000)
        grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000)
        grad_norm_ms_stftd = torch.nn.utils.clip_grad_norm_(ms_stftd.parameters(), 1000)
        optim_d.step()

        # MPD loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, fake_y)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

        # MRD loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, fake_y)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        # MRD loss
        y_ds_hat_r2, y_ds_hat_g2, fmap_s_r2, fmap_s_g2 = ms_stftd(y, fake_y)
        loss_fm_s2 = feature_loss(fmap_s_r2, fmap_s_g2)
        loss_gen_s2, losses_gen_s2 = generator_loss(y_ds_hat_g2)

        loss_gen_all = (
            loss_gen_s
            + loss_gen_f
            + loss_fm_s
            + loss_fm_f
            + loss_gen_s2
            + loss_fm_s2
            + loss_msmel * hps.train.w_stft
        )

        loss_gen_all.backward()
        grad_norm_g = commons.clip_grad_value_(model.parameters(), 1000)
        optimizer.step()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                losses = [loss_msmel]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "learning_rate": lr,
                    "grad_norm_g": grad_norm_g,
                    "grad_norm_mpd": grad_norm_mpd,
                    "grad_norm_mrd": grad_norm_mrd,
                    "grad_norm_ms_stftd": grad_norm_ms_stftd,
                }
                scalar_dict.update({
                    "loss/g/msmel": loss_msmel,
                    "loss/g/mpd_gen": loss_gen_f,
                    "loss/g/mrd_gen": loss_gen_s,
                    "loss/g/ms_stftd_gen": loss_gen_s2,
                    "loss/g/mpd_fm": loss_fm_f,
                    "loss/g/mrd_fm": loss_fm_s,
                    "loss/g/ms_stftd_fm": loss_fm_s2,
                })

                scalar_dict.update({
                    "gan/mpd_g/{}".format(i): v for i, v in enumerate(losses_gen_f)
                })
                scalar_dict.update({
                    "gan/mrd_g/{}".format(i): v for i, v in enumerate(losses_gen_s)
                })
                scalar_dict.update({
                    "gan/ms_stftd_g/{}".format(i): v
                    for i, v in enumerate(losses_gen_s2)
                })
                scalar_dict.update({
                    "gan/mpd_d_r/{}".format(i): v for i, v in enumerate(losses_disc_f_r)
                })
                scalar_dict.update({
                    "gan/mrd_d_r/{}".format(i): v for i, v in enumerate(losses_disc_s_r)
                })
                scalar_dict.update({
                    "gan/ms_stftd_d_r/{}".format(i): v
                    for i, v in enumerate(losses_disc_s_r2)
                })
                scalar_dict.update({
                    "gan/mpd_d_g/{}".format(i): v for i, v in enumerate(losses_disc_f_g)
                })
                scalar_dict.update({
                    "gan/mrd_d_g/{}".format(i): v for i, v in enumerate(losses_disc_s_g)
                })
                scalar_dict.update({
                    "gan/ms_stftd_d_g/{}".format(i): v
                    for i, v in enumerate(losses_disc_s_g2)
                })

                utils.summarize(
                    writer=writer, global_step=global_step, scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                torch.cuda.empty_cache()
                # evaluate(hps, model, eval_loader, writer, utmos_model)

            if global_step % hps.train.save_interval == 0:
                utils.save_checkpoint(
                    model,
                    optimizer,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    mpd,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "MPD_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    mrd,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "MRD_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    ms_stftd,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "MSTFTD_{}.pth".format(global_step)),
                )
        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, model, eval_loader, writer_eval, utmos_model):
    model.eval()
    image_dict = {}
    audio_dict = {}

    # modules for evaluation metrics
    pesq_resampler = torchaudio.transforms.Resample(
        hps.data.sampling_rate, 16000
    ).cuda()
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

    val_err_tot = 0
    val_pesq_tot = 0
    val_mrstft_tot = 0
    val_utmos_tot = 0

    with torch.no_grad():
        for batch_idx, (mel, y, target_std, _) in enumerate(eval_loader):
            y = y.cuda(0)
            mel = mel.cuda(0)
            target_std = target_std.cuda(0)

            y_gen = model(
                y,
                mel,
                target_std,
                n_timesteps=hps.train.tuning_steps,
                temperature=hps.train.finetuning_temperature,
            )

            if torch.abs(y_gen).max() >= 0.95:
                y_gen = (y_gen / (torch.abs(y_gen).max())) * 0.95

            y_gen_mel = mel_spectrogram(
                y_gen.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            val_err_tot += F.l1_loss(mel, y_gen_mel).item()

            y_16k = pesq_resampler(y)
            y_g_hat_16k = pesq_resampler(y_gen.squeeze(1))
            y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
            y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
            val_pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")

            # MRSTFT calculation
            val_mrstft_tot += loss_mrstft(y_gen, y).item()
            val_utmos_tot += utmos_model(y_g_hat_16k, 16000)

            if batch_idx <= 4:
                plot_mel = torch.cat([mel, y_gen_mel], dim=1)
                plot_mel = plot_mel.clip(min=-10, max=10)

                image_dict.update({
                    "gen/mel_{}".format(batch_idx): utils.plot_spectrogram_to_numpy(
                        plot_mel.squeeze().cpu().numpy()
                    ),
                })
                audio_dict.update({
                    "gen/audio_{}_gen".format(batch_idx): y_gen.squeeze(),
                })
                if global_step == 0:
                    audio_dict.update({"gt/audio_{}".format(batch_idx): y.squeeze()})

        val_err_tot /= batch_idx + 1
        val_pesq_tot /= batch_idx + 1
        val_mrstft_tot /= batch_idx + 1
        val_utmos_tot /= batch_idx + 1

    scalar_dict = {
        "val/mel": val_err_tot,
        "val/pesq": val_pesq_tot,
        "val/mrstft": val_mrstft_tot,
        "val/utmos": val_utmos_tot,
    }
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict,
    )
    model.train()


if __name__ == "__main__":
    main()

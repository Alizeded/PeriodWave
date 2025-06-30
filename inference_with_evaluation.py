import os
import torch
import argparse
import tqdm
import numpy as np
from torch.nn import functional as F
from scipy.io.wavfile import write
import torchaudio
import utils
from meldataset_prior_length import (
    mel_spectrogram,
    load_wav,
    MAX_WAV_VALUE,
    parse_filelist,
)
from librosa.util import normalize

import auraloss
from pesq import pesq
import torchcrepe
from Eval.pitch_periodicity import from_audio, p_p_F

from periodwave.periodwave import FlowMatch

h = None
device = None


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def inference(a):
    torch.manual_seed(1234)
    np.random.seed(1234)

    os.makedirs(a.output_dir, exist_ok=True)
    model = FlowMatch(
        hps.data.n_mel_channels, hps.model.periods, hps.model.noise_scale
    ).cuda()

    num_param = get_param_num(model)
    print("[Model] number of Parameters:", num_param)

    _ = model.eval()
    _ = utils.load_checkpoint(a.ckpt, model, None)

    model.estimator.remove_weight_norm()

    threshold = torchcrepe.threshold.Hysteresis()

    energy_max = float(np.load(hps.data.energy_max, allow_pickle=True))
    energy_min = float(np.load(hps.data.energy_min, allow_pickle=True))
    std_min = 0.1

    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).cuda()

    pesq_resampler = torchaudio.transforms.Resample(
        hps.data.sampling_rate, 16000
    ).cuda()
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

    wavs_test = parse_filelist(hps.data.test_filelist_path)

    i = 0

    mel_error = 0
    pesq_wb = 0
    pesq_nb = 0

    pitch_total = 0
    periodicity_total = 0
    f1_total = 0
    utmos = 0
    val_mrstft_tot = 0

    for source_path in tqdm.tqdm(wavs_test, desc="synthesizing each utterance"):
        audio, _ = load_wav(source_path, hps.data.sampling_rate)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        if (audio.size(1) % hps.data.hop_length) != 0:
            audio = audio[:, : -(audio.size(1) % hps.data.hop_length)]

        file_name = os.path.splitext(os.path.basename(source_path))[0]
        audio = audio.cuda()

        mel = mel_spectrogram(
            audio,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            center=False,
        )
        energy = (mel.exp()).sum(1).sqrt()
        target_std = torch.clamp(
            (energy - energy_min) / (energy_max - energy_min), std_min, None
        )
        target_std = torch.repeat_interleave(target_std, 256, dim=1)

        assert audio.shape[1] == mel.shape[2] * hps.data.hop_length, (
            "audio shape {} mel shape {}".format(audio.shape, mel.shape)
        )

        with torch.no_grad():
            resynthesis_audio = model(
                audio,
                mel,
                target_std.unsqueeze(0),
                n_timesteps=a.iter,
                temperature=a.noise_scale,
                solver=a.solver,
            )

            # resynthesis_audio = (resynthesis_audio / (torch.abs(resynthesis_audio).max())) * 0.95

            if torch.abs(resynthesis_audio).max() >= 0.95:
                resynthesis_audio = (
                    resynthesis_audio / (torch.abs(resynthesis_audio).max())
                ) * 0.95

            mel_hat = mel_spectrogram(
                resynthesis_audio.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
                center=False,
            )

            mel_error += F.l1_loss(mel, mel_hat).item()

            y_16k = pesq_resampler(audio)
            y_g_hat_16k = pesq_resampler(resynthesis_audio.squeeze(1))

            hopsize = int(256 * (torchcrepe.SAMPLE_RATE / 24000))
            padding = int((1024 - hopsize) // 2)

            audio_for_pitch = torch.nn.functional.pad(
                y_16k[None], (padding, padding), mode="reflect"
            ).squeeze(0)

            gen_audio_for_pitch = torch.nn.functional.pad(
                y_g_hat_16k[None], (padding, padding), mode="reflect"
            ).squeeze(0)

            ori_audio_len = audio.shape[-1] // 256
            true_pitch, true_periodicity = from_audio(
                audio_for_pitch.squeeze(), ori_audio_len, hopsize
            )
            fake_pitch, fake_periodicity = from_audio(
                gen_audio_for_pitch.squeeze(), ori_audio_len, hopsize
            )

            pitch, periodicity, f1 = p_p_F(
                threshold, true_pitch, true_periodicity, fake_pitch, fake_periodicity
            )

            pitch_total += pitch
            f1_total += f1

            periodicity_total += periodicity

            utmos += predictor(y_g_hat_16k, 16000)
            y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
            y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()

            pesq_wb += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")
            # pesq_nb += pesq(16000, y_int_16k, y_g_hat_int_16k, 'nb')

            # MRSTFT calculation
            val_mrstft_tot += loss_mrstft(resynthesis_audio, audio).item()

            resynthesis_audio = resynthesis_audio.squeeze()[: audio.shape[-1]]
            resynthesis_audio = resynthesis_audio * MAX_WAV_VALUE
            resynthesis_audio = resynthesis_audio.cpu().numpy().astype("int16")

            file_name = os.path.splitext(os.path.basename(source_path))[0]
            file_name = "{}.wav".format(file_name)

            output_file = os.path.join(
                "periodwave_turbo_libritts_dev"
                + "_"
                + str(a.solver)
                + "_"
                + str(a.iter)
                + "_"
                + str(a.noise_scale),
                file_name,
            )

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            write(output_file, 24000, resynthesis_audio)

        i += 1

    mel_error = mel_error / i
    pesq_wb = pesq_wb / i
    pitch_total = pitch_total / i
    periodicity_total = periodicity_total / i
    f1_total = f1_total / i
    utmos = utmos / i
    val_mrstft_tot = val_mrstft_tot / i

    with open(
        os.path.join(
            "periodwave_turbo_libritts_dev"
            + "_"
            + str(a.solver)
            + "_"
            + str(a.iter)
            + "_"
            + str(a.noise_scale),
            "score_list.txt",
        ),
        "w",
    ) as f:
        f.write(
            "periodwave_turbo_libritts_dev Solver:{}\nIter: {}\nNoise_scale: {}\n".format(
                a.solver, a.iter, a.noise_scale
            )
        )
        f.write("UTMOS: {}\n".format(utmos))
        f.write("Mel L1 distance: {}\nMR-STFT: {}\n".format(mel_error, val_mrstft_tot))
        f.write("PESQ Wide Band: {}\nPESQ Narrow Band {}\n".format(pesq_wb, pesq_nb))
        f.write(
            "Pitch: {}\nPeriodicity: {}\nV/UV F1: {}\n".format(
                pitch_total, periodicity_total, f1_total
            )
        )


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="gt")
    parser.add_argument("--output_dir", default="test")
    parser.add_argument(
        "--ckpt", default="logs/periodwave_turbo_4_msmel_45_mel_gan_2e5/G_274000.pth"
    )
    parser.add_argument("--iter", default=4, type=int)
    parser.add_argument("--noise_scale", default=1, type=float)
    parser.add_argument("--solver", default="euler", help="euler midpoint heun rk4")
    a = parser.parse_args()

    global hps, device
    hps = utils.get_hparams_from_file(
        os.path.join(os.path.split(a.ckpt)[0], "config.json")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(a)


if __name__ == "__main__":
    main()

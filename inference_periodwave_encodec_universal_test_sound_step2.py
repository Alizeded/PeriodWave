import os
import torch
import argparse
from glob import glob
import tqdm
import numpy as np
from torch.nn import functional as F
from scipy.io.wavfile import write
import torchaudio
import periodwave.inference_utils.utils as utils
from periodwave.inference_utils.meldataset_prior_length import (
    mel_spectrogram,
    load_wav,
    MAX_WAV_VALUE,
)
from librosa.util import normalize

import auraloss
from pesq import pesq

from periodwave.periodwave_encodec_freeu import FlowMatch
from periodwave.inference_utils.encodec_feature_extractor import EncodecFeatures

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
        hps.data.n_mel_channels,
        hps.model.periods,
        hps.model.noise_scale,
        hps.model.final_dim,
        hps.model.hidden_dim,
    ).cuda()

    num_param = get_param_num(model)
    print("[Model] number of Parameters:", num_param)

    _ = model.eval()
    _ = utils.load_checkpoint(a.ckpt, model, None)

    model.estimator.remove_weight_norm()

    Encodec = EncodecFeatures(bandwidth=a.bw).cuda()
    # 6.0 (Default, we trained the model with the feature of 6.0)
    # 1.5, 3.0, 6.0, 12.0
    # 12.0 (Not used during training but our model can generate higher quality audio with 12.0)

    pesq_resampler = torchaudio.transforms.Resample(
        hps.data.sampling_rate, 16000
    ).cuda()
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

    wavs_test = []
    wavs_test += sorted(
        glob(
            "audio_reconstruct_universal_testset_v2/sound_effect/**/*.wav",
            recursive=True,
        )
    )

    i = 0
    pitch_eval = False
    mel_error = 0
    pesq_wb = 0
    pesq_nb = 0

    val_mrstft_tot = 0
    mel_L = 0
    mel_M = 0
    mel_H = 0
    pitch_count = 0
    pseq_error = 0

    for source_path in tqdm.tqdm(wavs_test, desc="synthesizing each utterance"):
        audio, _ = load_wav(source_path, hps.data.sampling_rate)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        audio = F.pad(
            audio, (0, ((audio.size(1) // 3840) + 1) * 3840 - audio.size(1)), "constant"
        )

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

        assert audio.shape[1] == mel.shape[2] * hps.data.hop_length, (
            "audio shape {} mel shape {}".format(audio.shape, mel.shape)
        )

        with torch.no_grad():
            embs = Encodec(audio)
            resynthesis_audio = model(
                audio,
                embs,
                n_timesteps=a.iter,
                temperature=a.noise_scale,
                solver=a.solver,
                sway=a.sway,
                sway_coef=a.sway_coef,
                s_w=a.s_w,
                b_w=a.b_w,
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

            mel_L += F.l1_loss(mel[:, :61, :], mel_hat[:, :61, :]).item()
            mel_M += F.l1_loss(mel[:, 60:81, :], mel_hat[:, 60:81, :]).item()
            mel_H += F.l1_loss(mel[:, 80:100, :], mel_hat[:, 80:100, :]).item()

            y_16k = pesq_resampler(audio)
            y_g_hat_16k = pesq_resampler(resynthesis_audio.squeeze(1))

            y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
            y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()

            try:
                pesq_wb += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")
            except:
                pseq_error += 1
            # pesq_nb += pesq(16000, y_int_16k, y_g_hat_int_16k, 'nb')

            # MRSTFT calculation
            val_mrstft_tot += loss_mrstft(resynthesis_audio, audio).item()

            resynthesis_audio = resynthesis_audio.squeeze()[: audio.shape[-1]]
            resynthesis_audio = resynthesis_audio * MAX_WAV_VALUE
            resynthesis_audio = resynthesis_audio.cpu().numpy().astype("int16")

            file_name = os.path.splitext(os.path.basename(source_path))[0]
            file_name = "{}.wav".format(file_name)

            output_file = os.path.join(
                "periodwave_encodec_base_turbo_step2_400k_rfwave_sound"
                + "_"
                + str(a.solver)
                + "_"
                + str(a.bw)
                + "_"
                + str(a.iter)
                + "_"
                + str(a.noise_scale)
                + "_"
                + str(a.s_w)
                + "_"
                + str(a.b_w)
                + "_sway_"
                + str(a.sway),
                file_name,
            )

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            write(output_file, 24000, resynthesis_audio)

        i += 1

    mel_error = mel_error / i
    pesq_wb = pesq_wb / (i - pseq_error)
    val_mrstft_tot = val_mrstft_tot / i
    mel_L = mel_L / i
    mel_M = mel_M / i
    mel_H = mel_H / i

    with open(
        os.path.join(
            "periodwave_encodec_base_turbo_step2_400k_rfwave_sound"
            + "_"
            + str(a.solver)
            + "_"
            + str(a.bw)
            + "_"
            + str(a.iter)
            + "_"
            + str(a.noise_scale)
            + "_"
            + str(a.s_w)
            + "_"
            + str(a.b_w)
            + "_sway_"
            + str(a.sway),
            "score_list.txt",
        ),
        "w",
    ) as f:
        f.write(
            "periodwave_encodec_base_turbo_step2_400k_rfwave_sound Solver:{}\nIter: {}\nNoise_scale: {}\n".format(
                a.solver, a.iter, a.noise_scale
            )
        )
        f.write("bw: {}\n".format(a.bw))
        f.write("s_w: {}\nb_w: {}\n".format(a.s_w, a.b_w))
        f.write("Sway: {}\n".format(a.sway))
        f.write("Mel L1 distance: {}\nMR-STFT: {}\n".format(mel_error, val_mrstft_tot))
        f.write("PESQ Wide Band: {}\nPESQ Narrow Band {}\n".format(pesq_wb, pesq_nb))
        f.write("mel_L: {}\nmel_M: {}\nmel_H: {}\n".format(mel_L, mel_M, mel_H))


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="gt")
    parser.add_argument("--output_dir", default="test")
    parser.add_argument(
        "--ckpt",
        default="logs/periodwave_encodec_turbo_universe_cont_step2/G_400000.pth",
    )
    parser.add_argument("--bw", default=6.0, type=float)
    parser.add_argument("--iter", default=2, type=int)
    parser.add_argument("--noise_scale", default=1, type=float)
    parser.add_argument("--solver", default="euler", help="euler midpoint heun rk4")
    parser.add_argument("--s_w", default=1, type=float)
    parser.add_argument("--b_w", default=1, type=float)
    parser.add_argument("--sway", default=False, type=bool)
    parser.add_argument("--sway_coef", default=-1.0, type=float)
    a = parser.parse_args()

    global hps, device
    hps = utils.get_hparams_from_file(
        os.path.join(os.path.split(a.ckpt)[0], "config.json")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(a)


if __name__ == "__main__":
    main()

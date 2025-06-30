import os
import torch
import argparse
import tqdm
import numpy as np
from scipy.io.wavfile import write
import utils
from meldataset_prior_length import (
    mel_spectrogram,
    load_wav,
    MAX_WAV_VALUE,
    parse_filelist,
)
from librosa.util import normalize
from periodwave.periodwave_large import FlowMatch

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

    energy_max = float(np.load(hps.data.energy_max, allow_pickle=True))
    energy_min = float(np.load(hps.data.energy_min, allow_pickle=True))
    std_min = 0.1

    wavs_test = parse_filelist(hps.data.test_filelist_path)

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

            if torch.abs(resynthesis_audio).max() >= 0.95:
                resynthesis_audio = (
                    resynthesis_audio / (torch.abs(resynthesis_audio).max())
                ) * 0.95

            resynthesis_audio = resynthesis_audio.squeeze()[: audio.shape[-1]]
            resynthesis_audio = resynthesis_audio * MAX_WAV_VALUE
            resynthesis_audio = resynthesis_audio.cpu().numpy().astype("int16")

            file_name = os.path.splitext(os.path.basename(source_path))[0]
            file_name = "{}.wav".format(file_name)

            output_file = os.path.join(
                "periodwave_turbo_large"
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


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="gt")
    parser.add_argument("--output_dir", default="test")
    parser.add_argument("--ckpt", default="logs/periodwave_turbo_8_large/G_379000.pth")
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

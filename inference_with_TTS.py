import os
import torch
import argparse
from glob import glob
import tqdm
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import periodwave.inference_utils.utils as utils
from periodwave.inference_utils.meldataset_prior_length import MAX_WAV_VALUE


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

    energy_max = float(np.load(hps.data.energy_max, allow_pickle=True))
    energy_min = float(np.load(hps.data.energy_min, allow_pickle=True))
    std_min = 0.1

    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).cuda()

    pesq_resampler = torchaudio.transforms.Resample(
        hps.data.sampling_rate, 16000
    ).cuda()

    mels = glob("ardit_dmd_b1/*.mel", recursive=True)

    i = 0
    utmos = 0

    for mel_path in tqdm.tqdm(mels, desc="synthesizing each utterance"):
        file_name = os.path.splitext(os.path.basename(mel_path))[0]
        mel = torch.load(mel_path)
        mel = torch.FloatTensor(mel).cuda()
        mel = mel.transpose(1, 2)
        audio = torch.zeros((1, mel.shape[-1] * 256)).cuda()

        energy = (mel.exp()).sum(1).sqrt()
        target_std = torch.clamp(
            (energy - energy_min) / (energy_max - energy_min), std_min, None
        )
        target_std = torch.repeat_interleave(target_std, 256, dim=1)

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

            y_g_hat_16k = pesq_resampler(resynthesis_audio.squeeze(1))
            utmos += predictor(y_g_hat_16k, 16000)

            resynthesis_audio = resynthesis_audio.squeeze()[: audio.shape[-1]]
            resynthesis_audio = resynthesis_audio * MAX_WAV_VALUE
            resynthesis_audio = resynthesis_audio.cpu().numpy().astype("int16")

            file_name = os.path.splitext(os.path.basename(mel_path))[0]
            file_name = "{}.wav".format(file_name)

            output_file = os.path.join(
                "periodwave_ardit"
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
    utmos = utmos / i

    with open(
        os.path.join(
            "periodwave_ardit"
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
            "periodwave_ardit Solver:{}\nIter: {}\nNoise_scale: {}\n".format(
                a.solver, a.iter, a.noise_scale
            )
        )
        f.write("UTMOS: {}\n".format(utmos))


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

import glob
import argparse
import torchaudio
import torch
import tqdm
import os

def main(args):

##############################
    

    path = args.input_dir
    wavs_train = []
    wavs_train += sorted(glob.glob(path+'/train-clean-100/**/*.wav', recursive=True))
    wavs_train += sorted(glob.glob(path+'/train-clean-360/**/*.wav', recursive=True))
    wavs_train += sorted(glob.glob(path+'/train-other-500/**/*.wav', recursive=True))

    with open('filelists_24k/train_wav.txt', 'w') as f:
        for wav in wavs_train:
            f.write(wav+'\n')


    with open(args.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(path, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    with open(args.input_validation_file2, 'r', encoding='utf-8') as fi:
        validation_files += [os.path.join(path, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
        print("first validation file: {}".format(validation_files[0]))

    with open('filelists_24k/val_wav.txt', 'w') as f:
        for wav in validation_files:
            f.write(wav + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='./LibriTTS')
    parser.add_argument('--input_validation_file', default='./filelists_24k/dev-clean.txt')
    parser.add_argument('--input_validation_file2', default='./filelists_24k/dev-other.txt')
    a = parser.parse_args()

    main(a)
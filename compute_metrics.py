# There are two files are used to compute the metrics:
# 1. Original audio file: At ./dataset/testset/gt_wavs
# 2. Upsampled (generated) audio file: At ./test_samples/48000/8000 (or 12000, 16000, 24000)
# We parse those files and compute the metrics

import os
import csv
import argparse
import torch
import torch.nn as nn
import torchaudio
import glob
from tqdm import tqdm
from time import time


def save_results_to_csv(results, filename="results.csv"):
    # Check if file exists. If not, create it and write headers
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Sample Rate",
                    "SNR (mean)",
                    "SNR (std)",
                    "LSD (mean)",
                    "LSD (std)",
                    "LSD HF (mean)",
                    "LSD HF (std)",
                    "LSD LF (mean)",
                    "LSD LF (std)",
                ]
            )

        writer.writerow(results)
        print(f"Results saved to {filename}")


class STFTMag(nn.Module):
    def __init__(self, nfft=1024, hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(nfft), False)

    # x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        self.window = self.window.to(x.device)
        stft = torch.stft(
            x, self.nfft, self.hop, window=self.window, return_complex=True
        )  # [B, F, TT]
        #   return_complex=False)  #[B, F, TT,2]
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2))
        return mag


def main(args):
    print(f"Running {args.input_sr} to {args.target_sr} | CUDA: {args.cuda}")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    stft = STFTMag(2048, 512).to(device)
    hf = int(1025 * (args.input_sr / args.target_sr))
    output_path = os.path.join(args.output_dir, str(args.target_sr), str(args.input_sr))
    # Make sure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    def cal_snr(pred, target):
        return (
            20
            * torch.log10(
                torch.norm(target, dim=-1)
                / torch.norm(pred - target, dim=-1).clamp(min=1e-8)
            )
        ).mean()

    def cal_lsd(pred, target, hf):
        sp = torch.log10(stft(pred).square().clamp(1e-8))
        st = torch.log10(stft(target).square().clamp(1e-8))
        return (
            (sp - st).square().mean(dim=1).sqrt().mean(),
            (sp[:, hf:, :] - st[:, hf:, :]).square().mean(dim=1).sqrt().mean(),
            (sp[:, :hf, :] - st[:, :hf, :]).square().mean(dim=1).sqrt().mean(),
        )

    # Get the list of original audio files
    gt_files = sorted(glob.glob("./dataset/testset/gt_wavs/*.wav"))
    # Get the list of upsampled audio files
    gen_files = sorted(
        glob.glob(f"{args.source_dir}/{args.target_sr}/{args.input_sr}/*.wav")
    )

    # Check if the order and filenames are correct
    assert len(gt_files) == len(gen_files)
    for i in range(len(gt_files)):
        assert os.path.basename(gt_files[i]) == os.path.basename(gen_files[i])

    print(f"Total files: {len(gt_files)}")

    results = []
    for i in range(1):
        # Compute the metrics
        snr_list = []
        lsd_list = []
        lsd_hf_list = []
        lsd_lf_list = []

        for j, (gt_file, gen_file) in enumerate(
            tqdm(zip(gt_files, gen_files), total=len(gt_files))
        ):
            # Load the audio files
            gt, _ = torchaudio.load(gt_file)
            gen, _ = torchaudio.load(gen_file)

            # Move to device
            gt = gt.to(device)
            gen = gen.to(device)

            start_time = time()
            # Compute the metrics
            snr = cal_snr(gen, gt)
            lsd, lsd_hf, lsd_lf = cal_lsd(gen, gt, hf)

            snr_list.append(snr)
            lsd_list.append(lsd)
            lsd_hf_list.append(lsd_hf)
            lsd_lf_list.append(lsd_lf)

            if i == 0:
                base = os.path.basename(gen_file).replace(".wav", "")
                gen_out = os.path.join(output_path, f"{base}_up.wav")
                gt_out = os.path.join(output_path, f"{base}_orig.wav")
                # Copy the audio files to the output directory
                os.system(f"cp {gen_file} {gen_out}")
                os.system(f"cp {gt_file} {gt_out}")

            snr = torch.stack(snr_list, dim=0).mean()
            lsd = torch.stack(lsd_list, dim=0).mean()
            lsd_hf = torch.stack(lsd_hf_list, dim=0).mean()
            lsd_lf = torch.stack(lsd_lf_list, dim=0).mean()

            dict = {
                "snr": f"{snr.item():.2f}",
                "lsd": f"{lsd.item():.2f}",
                "lsd_hf": f"{lsd_hf.item():.2f}",
                "lsd_lf": f"{lsd_lf.item():.2f}",
            }

            results.append(
                torch.stack(
                    [
                        snr.cpu(),
                        lsd.cpu(),
                        lsd_hf.cpu(),
                        lsd_lf.cpu(),
                    ],
                    dim=0,
                ).unsqueeze(-1)
            )
        print(dict)

    # Save results to csv
    # Loop the results and calculate mean and std of results
    results = torch.cat(results, dim=1)
    # Get mean and std in [[mean, std], [mean, std], ...] format
    results = torch.stack([results.mean(dim=1), results.std(dim=1)], dim=1)
    # Convert to [mean, std, mean, std]
    results = results.flatten().tolist()
    # Add sample rate to the beginning of the list
    results.insert(0, args.input_sr)

    save_results_to_csv(results, "results_48kHz.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input sample rate
    parser.add_argument(
        "-sr", "--input_sr", type=int, required=True, help="Input sample rate"
    )
    # Target sample rate
    parser.add_argument(
        "-tsr", "--target_sr", type=int, required=True, help="Target sample rate"
    )
    # Source directory
    parser.add_argument("-src", "--source_dir", default="./outputs/")
    # Output directory
    parser.add_argument("-out", "--output_dir", default="./test_samples/")
    # Enable CUDA
    parser.add_argument("-cuda", "--cuda", type=bool, default=False)
    args = parser.parse_args()
    main(args)

    # Example to run the script:
    # python compute_metrics.py -sr 8000 -tsr 48000

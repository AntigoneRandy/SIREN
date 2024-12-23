import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import argparse
import random
import datetime
np.random.seed(777)

def calculate_nth_term(n):
    if n % 2 == 1:
        return 10**(-((n+1)//2))
    else:
        return 10**(-(n//2)) * 0.5

# Set up command line arguments
parser = argparse.ArgumentParser(description='Process clean and watermark samples paths, and output filename.')
parser.add_argument('--clean_path', type=str, help='The path to clean samples')
parser.add_argument('--coating_path', type=str, help='The path to watermark samples')
parser.add_argument('--output', type=str, default='', help='The output filename (optional)')
parser.add_argument('--repeat', type=int, default=100000, help='Number of sampling times')
parser.add_argument('--samples', type=int, default=10, help='Number of samples selected for each test')

# Parse command line arguments
args = parser.parse_args()
comments = input("Enter experimenter + base model + dataset + fine-tuning method + watermarking method + additional notes\n")

# Read data
clean_samples_path = args.clean_path
watermark_samples_path = args.coating_path
clean_samples_df = pd.read_csv(clean_samples_path, delimiter=',', header=None)
watermark_samples_df = pd.read_csv(watermark_samples_path, delimiter=',', header=None)

# Data processing
clean_samples = clean_samples_df.apply(pd.to_numeric, errors='coerce').dropna().values.flatten()
watermark_samples = watermark_samples_df.apply(pd.to_numeric, errors='coerce').dropna().values.flatten()

# Perform N times resampling and KS test
N = args.repeat
clean_samples = np.random.choice(clean_samples, 300, replace=False)
p_values = []
for _ in range(N):
    sampled_watermark = np.random.choice(watermark_samples, args.samples, replace=False)
    ks_stat, p_value = ks_2samp(sampled_watermark, clean_samples[:300], alternative="greater")
    p_values.append(p_value)

# Count p-values under different thresholds
thresholds = [calculate_nth_term(n) for n in range(1, 40)]
counts = {threshold: sum(np.array(p_values) < threshold) / N for threshold in thresholds}
print(counts)

# Prepare log content
log_content = [
    "Experimenter + base model + dataset + fine-tuning method + watermarking method + additional notes",
    comments,
    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Command: python ks_test_final.py --clean_path {args.clean_path} --coating_path {args.coating_path} --output {args.output} --repeat {args.repeat} --samples {args.samples}",
    f"Clean_sample_size: {clean_samples.size}, Watermark_sample_size: {watermark_samples.size}",
    f"Clean statistics: ({clean_samples.mean()}, {clean_samples.var()})",
    f"Watermark statistics: ({watermark_samples.mean()}, {watermark_samples.var()})",
    "",
    "Threshold:",
    str(counts)
]

# Check output filename and write to log
output_filename = args.output if args.output else "output.log"
with open(output_filename, 'w') as log_file:
    for line in log_content:
        log_file.write(line + "\n")
print(f"Task completed, results have been saved to {output_filename} file.")

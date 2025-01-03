import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', help='List of weights to merge. Example of using --weights argument: python merge_weights.py --weights weight1.bin weight2.bin weight3.bin', required=True, default=[])
parser.add_argument('--output', help='Output file name for the merged weights. The full filepath including extension', required=True, default='./merged_weights.bin')
args = parser.parse_args()

weights = args.weights
assert len(weights) > 1, "Please provide more than one weight file to merge."

# Load the weights
loaded_weights = []
for weight in weights:
    loaded_weights.append(torch.load(weight, map_location='cpu'))

# Merge the weights into a single dictionary
merged_weights = {}
for i, weight in enumerate(loaded_weights):
    for key in weight.keys():
        if key in merged_weights:
            print(f"Key '{key}' already exists in the merged weights. Overwriting the value.")

        merged_weights[key] = weight[key]

# Save the merged weights
torch.save(merged_weights, args.output)
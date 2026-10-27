from argparse import ArgumentParser

import torch
from dataset_utils import get_num_items, get_test_dataloader
from eval_utils import evaluate
from utils import build_model, get_device, load_config

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--disable_ffn_layers', type=int, nargs='*', default=[], help='Indices of Transformer layers to disable FFN for (e.g. --disable_ffn_layers 0 1)')
parser.add_argument('--disable_l1_ffn', action='store_true', help='Convenience alias to disable Layer 1 FFN')
parser.add_argument('--bias_only_ffn_layers', type=int, nargs='*', default=[], help='Indices of Transformer layers to set FFN to bias-only mode')
parser.add_argument('--bias_only_l1_ffn', action='store_true', help='Convenience alias to set Layer 1 FFN to bias-only mode')
args = parser.parse_args()

disabled_layers = list(args.disable_ffn_layers)
if args.disable_l1_ffn and 1 not in disabled_layers:
    disabled_layers.append(1)

bias_only_layers = list(args.bias_only_ffn_layers)
if args.bias_only_l1_ffn and 1 not in bias_only_layers:
    bias_only_layers.append(1)

config = load_config(args.config)
num_items = get_num_items(config.dataset_name) 
device = get_device()
model = build_model(config)
model = model.to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))

if disabled_layers:
    for layer_idx in disabled_layers:
        model.disable_layer_ffn(layer_idx)
    print(f"FFN DISABLED for layers: {disabled_layers}")

if bias_only_layers:
    for layer_idx in bias_only_layers:
        model.set_layer_ffn_bias_only(layer_idx)
    print(f"FFN BIAS-ONLY for layers: {bias_only_layers}")

test_dataloader = get_test_dataloader(config.dataset_name, batch_size=config.eval_batch_size, max_length=config.sequence_length)
evaluation_result = evaluate(model, test_dataloader, config.metrics, config.recommendation_limit, 
                                 config.filter_rated, device=device) 
print(evaluation_result)

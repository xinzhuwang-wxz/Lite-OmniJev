"""No-weight compatibility probe for the vLLM 0.26 Gemma4 config boundary."""

import argparse

from transformers import AutoConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model_dir")
args = parser.parse_args()
config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
text = config.text_config
# vLLM's Gemma4 converter must read both dimensions to allocate heterogeneous attention.
assert config.model_type == "gemma4"
assert max(text.head_dim, text.global_head_dim) > 0
print({"status": "passed", "head_dim": text.head_dim, "global_head_dim": text.global_head_dim})

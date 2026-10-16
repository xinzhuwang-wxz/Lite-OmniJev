"""Offline PEFT loading example only; not executed or a decision-readout benchmark."""

import argparse

from peft import PeftModel
from transformers import AutoModelForImageTextToText

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-local", required=True)
    parser.add_argument("--adapter-local", required=True)
    args = parser.parse_args()
    # Verify release-manifest hashes first. No online downloads or trust_remote_code.
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_local,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="bfloat16",
    )
    model = PeftModel.from_pretrained(base, args.adapter_local, local_files_only=True)
    model.eval()
    print("Adapter loaded. Use accepted project preparation/gateway for decision probabilities.")

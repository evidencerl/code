#!/usr/bin/env python3
"""Inference demo for Evidence-RL Qwen3.5-9B.

Usage:
    # Use released checkpoint from Hugging Face:
    python demo.py --image example.jpg --prompt "How many dogs are in the image?"

    # Or use local merged checkpoint:
    python demo.py --model-path checkpoints/qwen35_9b_answer_ced_ddp/merged_hf --image example.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-RL Inference Demo")
    parser.add_argument(
        "--model-path",
        default="hhj-ai/Evidence-RL-9B",
        help="Path or HuggingFace repo ID of the model (default: hhj-ai/Evidence-RL-9B)",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Path to input image file (optional, generates a blank test image if omitted)",
    )
    parser.add_argument(
        "--prompt",
        default="Look at the image carefully and describe the key visual evidence, then answer the question.",
        help="User text prompt / question",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load model on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generation tokens",
    )
    args = parser.parse_args()

    print(f"[demo] Loading processor and model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()

    if args.image and Path(args.image).is_file():
        image = Image.open(args.image).convert("RGB")
    else:
        print("[demo] No image provided; creating dummy 384x384 test image...")
        image = Image.new("RGB", (384, 384), color=(73, 109, 137))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    print(f"[demo] Prompt: {args.prompt}")
    print("[demo] Generating response...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    response = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("\n" + "=" * 40 + " Output " + "=" * 40)
    print(response.strip())
    print("=" * 88)


if __name__ == "__main__":
    main()

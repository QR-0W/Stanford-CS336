#!/usr/bin/env python3
"""
Text Generation Script for TinyStories Transformer

Loads a trained model checkpoint and generates text based on prompt.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import generate as generate_tokens


def load_model(checkpoint_path, device, args):
    print(f"Loading model from {checkpoint_path}...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, default="output/tiny_stories_vocab.json")
    parser.add_argument("--merges_file", type=str, default="output/tiny_stories_merges.json")

    # Model config (must match training)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Generation args
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)

    # 2. Load Model
    model = load_model(args.checkpoint, args.device, args)

    # 3. Generate
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"Prompt: {args.prompt}")
    print(f"Params: Temp={args.temperature}, Top-P={args.top_p}")
    print("=" * 60)

    prompt_ids = tokenizer.encode(args.prompt)

    for i in range(args.num_samples):
        generated_ids = generate_tokens(
            model=model,
            prompt_tokens=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        text = tokenizer.decode(generated_ids)
        print(f"\n[Sample {i + 1}]:")
        print("-" * 20)
        print(text)
        print("-" * 20)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Balanced NER Synthetic Data Generation - Command Line Interface

Uses object-oriented class design to generate synthetic data with balanced entity distribution
"""

import argparse
from pathlib import Path

from synthetic_data_generator import (
    create_balanced_dataset,
    create_topn_dataset,
    create_uniform_multi_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate NER synthetic data with balanced entity distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Generate by target sample count (auto-calculate top_n and samples_per_entity)
  python3 generate_balanced.py \\
      --source dataset/yakusha_annotated_data_few-shot \\
      --output dataset/my_balanced_data \\
      --target-samples 1500

  # Mode 2: Specify top_n and total samples (distribute by frequency ratio)
  python3 generate_balanced.py \\
      --source dataset/yakusha_annotated_data_few-shot \\
      --output dataset/yakusha_synthetic_topn20_uniform_chatgpt \\
      --top-n 20 \\
      --total-samples 10000

  # Mode 3: Type-level uniform sampling + multi-entity generation (recommended for low-frequency entity recognition)
  python3 generate_balanced.py \\
      --source dataset/yakusha_annotated_data_few-shot \\
      --output dataset/yakusha_synthetic_uniform_multi \\
      --uniform-multi \\
      --total-samples 10000 \\
      --entities-per-sample 3

  # Mode 4: Top-N + type-level uniform sampling + multi-entity generation (recommended)
  python3 generate_balanced.py \\
      --source dataset/yakusha_annotated_data_few-shot \\
      --output dataset/yakusha_synthetic_topn_uniform \\
      --uniform-multi \\
      --top-n 20 \\
      --total-samples 2000 \\
      --entities-per-sample 3

  # Generate using Claude API
  python3 generate_balanced.py \\
      --source dataset/yakusha_annotated_data_few-shot \\
      --output dataset/yakusha_synthetic_claude_uniform \\
      --uniform-multi \\
      --top-n 20 \\
      --total-samples 2000 \\
      --provider claude \\
      --model claude-haiku-4-5-20251001
        """
    )

    # Basic parameters
    parser.add_argument(
        '--source',
        default='../../dataset/yakusha_annotated_data_few-shot',
        help='Path to real dataset (used for distribution analysis)'
    )

    parser.add_argument(
        '--output',
        default='dataset/yakusha_synthetic_balanced',
        help='Output directory path'
    )

    # Generation parameters - Mode 1: By target sample count
    parser.add_argument(
        '--target-samples',
        type=int,
        default=None,
        help='Target total sample count (mutually exclusive with --top-n/--samples-per-entity)'
    )

    # Generation parameters - Mode 2: By top_n and total_samples
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Select top N entities with highest frequency from all entities'
    )

    parser.add_argument(
        '--total-samples',
        type=int,
        default=None,
        help='Target total sample count, distributed to each entity by frequency ratio'
    )

    # Generation parameters - Mode 3: Type-level uniform sampling + multi-entity generation
    parser.add_argument(
        '--uniform-multi',
        action='store_true',
        help='Use type-level uniform sampling + multi-entity generation mode (recommended for low-frequency entity recognition)'
    )

    parser.add_argument(
        '--entities-per-sample',
        type=int,
        default=3,
        help='Number of entities per sample (default: 3, matching test set distribution)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel threads (default: 10, recommended 10-20)'
    )

    # API parameters
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (reads from config file if not specified)'
    )

    parser.add_argument(
        '--model',
        default=None,
        help='Model to use (OpenAI default: gpt-4o-mini, Claude default: claude-3-haiku-20240307)'
    )

    parser.add_argument(
        '--provider',
        default='openai',
        choices=['openai', 'claude'],
        help='API provider (default: openai, options: claude)'
    )

    parser.add_argument(
        '--lang',
        default='ja',
        choices=['ja', 'de', 'fr', 'en'],
        help='Target language for prompt generation (default: ja)'
    )

    args = parser.parse_args()

    # Set default model based on provider
    if args.model is None:
        if args.provider == 'claude':
            args.model = 'claude-3-haiku-20240307'
        else:
            args.model = 'gpt-4o-mini'

    # Determine generation mode
    # uniform-multi mode has highest priority (can be combined with top_n)
    use_uniform_multi = args.uniform_multi
    # Only use pure top-n mode when uniform-multi is not used
    use_topn_mode = not use_uniform_multi and args.top_n is not None and args.total_samples is not None

    print("=" * 80)
    print("NER Synthetic Data Generation")
    print("=" * 80)
    print(f"Source dataset: {args.source}")
    print(f"Output directory: {args.output}")

    if use_uniform_multi:
        total_samples = args.total_samples or 10000
        if args.top_n is not None:
            print(f"Generation mode: Top-N + type-level uniform sampling + multi-entity generation")
            print(f"Top-N: {args.top_n}")
        else:
            print(f"Generation mode: Type-level uniform sampling + multi-entity generation (all entity types)")
        print(f"Target total samples: {total_samples}")
        print(f"Entities per sample: {args.entities_per_sample}")
    elif use_topn_mode:
        print(f"Generation mode: Global Top-N (distribute by frequency ratio)")
        print(f"Top-N: {args.top_n}")
        print(f"Target total samples: {args.total_samples}")
    else:
        target_samples = args.target_samples or 2000
        print(f"Generation mode: Distribute by entity type ratio")
        print(f"Target samples: {target_samples}")

    print(f"Parallel threads: {args.workers}")
    print(f"API provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Language: {args.lang}")
    print("=" * 80)

    # Generate data
    if use_uniform_multi:
        total_samples = args.total_samples or 10000
        dataset = create_uniform_multi_dataset(
            real_dataset_path=args.source,
            output_dir=args.output,
            total_samples=total_samples,
            entities_per_sample=args.entities_per_sample,
            top_n=args.top_n,
            api_key=args.api_key,
            model=args.model,
            num_workers=args.workers,
            provider=args.provider,
            lang=args.lang
        )
    elif use_topn_mode:
        dataset = create_topn_dataset(
            real_dataset_path=args.source,
            output_dir=args.output,
            top_n=args.top_n,
            total_samples=args.total_samples,
            api_key=args.api_key,
            model=args.model,
            num_workers=args.workers,
            provider=args.provider,
            lang=args.lang
        )
    else:
        target_samples = args.target_samples or 2000
        dataset = create_balanced_dataset(
            real_dataset_path=args.source,
            output_dir=args.output,
            target_samples=target_samples,
            api_key=args.api_key,
            model=args.model,
            num_workers=args.workers,
            lang=args.lang
        )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Generated data saved to: {args.output}/generated_ner_balanced.json")
    print("\nNext steps:")
    print("  1. Convert to Hugging Face datasets format")
    print("  2. Configure training experiment")
    print("  3. Run comparison experiments")
    print("=" * 80)


if __name__ == "__main__":
    main()

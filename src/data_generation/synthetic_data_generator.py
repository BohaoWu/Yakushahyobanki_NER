#!/usr/bin/env python3
"""
NER Synthetic Data Generator

Uses NERSyntheticDataGenerator class to coordinate the entire generation pipeline
"""

from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import time

try:
    from .synthetic_data_classes import (
        NERSyntheticDataCorpus,
        NERSyntheticDataModel,
        NERSyntheticDataDataset,
        NERSample
    )
except ImportError:
    from synthetic_data_classes import (
        NERSyntheticDataCorpus,
        NERSyntheticDataModel,
        NERSyntheticDataDataset,
        NERSample
    )


class NERSyntheticDataGenerator:
    """
    Synthetic Data Generator

    Features:
    1. Load existing dataset and analyze distribution
    2. Configure generation strategy based on real data distribution
    3. Use stratified sampling to generate balanced data
    4. Save results to specified path
    """

    def __init__(
        self,
        corpus: NERSyntheticDataCorpus,
        model: NERSyntheticDataModel,
        output_dir: str = "dataset"
    ):
        """
        Initialize generator

        Args:
            corpus: Corpus (real data)
            model: Generation model
            output_dir: Output directory
        """
        self.corpus = corpus
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_generation_config(
        self,
        target_total_samples: int = 2000,
        avg_entities_per_sample: int = 4,
        fixed_top_n: Optional[int] = None,
        fixed_samples_per_entity: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Calculate balanced generation configuration

        Args:
            target_total_samples: Target total sample count (ignored when both fixed_top_n and fixed_samples_per_entity are specified)
            avg_entities_per_sample: Average entities per sample
            fixed_top_n: Fixed top_n value (if specified, all entity types use the same value)
            fixed_samples_per_entity: Fixed samples per entity (if specified, all entity types use the same value)

        Returns:
            {entity_type: {top_n, samples_per_entity, ...}}
        """
        config = {}

        # If both fixed_top_n and fixed_samples_per_entity are specified, use fixed config
        use_fixed_config = fixed_top_n is not None and fixed_samples_per_entity is not None

        for entity_type, stats in self.corpus.entity_distribution.items():
            ratio = stats['ratio']
            unique_entities = stats['unique_entities']

            if use_fixed_config:
                # Use fixed config
                top_n = min(fixed_top_n, unique_entities)
                samples_per_entity = fixed_samples_per_entity
                target_samples = top_n * samples_per_entity
            else:
                # Calculate target entity count and sample count
                target_entities = int(target_total_samples * avg_entities_per_sample * ratio)
                target_samples = max(1, target_entities // avg_entities_per_sample)

                # If fixed_top_n is specified, use it; otherwise calculate based on target sample count
                if fixed_top_n is not None:
                    top_n = min(fixed_top_n, unique_entities)
                    samples_per_entity = max(1, target_samples // top_n)
                elif fixed_samples_per_entity is not None:
                    # Calculate top_n based on target_samples and fixed_samples_per_entity
                    top_n = min(max(1, target_samples // fixed_samples_per_entity), unique_entities)
                    samples_per_entity = fixed_samples_per_entity
                else:
                    # Auto-calculate top_n and samples_per_entity based on target sample count
                    if target_samples >= 150:
                        top_n = min(30, unique_entities)
                        samples_per_entity = max(1, target_samples // top_n)
                    elif target_samples >= 80:
                        top_n = min(20, unique_entities)
                        samples_per_entity = max(1, target_samples // top_n)
                    elif target_samples >= 40:
                        top_n = min(15, unique_entities)
                        samples_per_entity = max(1, target_samples // top_n)
                    elif target_samples >= 15:
                        top_n = min(10, unique_entities)
                        samples_per_entity = max(1, target_samples // top_n)
                    elif target_samples >= 5:
                        top_n = min(5, unique_entities)
                        samples_per_entity = max(1, target_samples // top_n)
                    else:
                        top_n = min(3, unique_entities)
                        samples_per_entity = max(1, target_samples // max(1, top_n))

            config[entity_type] = {
                'ratio': ratio,
                'unique_entities': unique_entities,
                'target_samples': target_samples,
                'top_n': top_n,
                'samples_per_entity': samples_per_entity,
                'estimated_samples': top_n * samples_per_entity
            }

        return config

    def print_generation_config(self, config: Dict[str, Dict]):
        """Print generation configuration"""
        print(f"\n{'='*80}")
        print(f"Balanced Data Generation Configuration")
        print(f"{'='*80}")
        print(f"\n{'Entity Type':<15} {'Top-N':>6} {'Samples/Entity':>14} {'Est. Samples':>12} {'Target Ratio':>12}")
        print('-' * 80)

        total_estimated = 0
        for entity_type, cfg in sorted(config.items(),
                                         key=lambda x: x[1]['ratio'], reverse=True):
            print(f"{entity_type:<15} {cfg['top_n']:>6} {cfg['samples_per_entity']:>14} "
                  f"{cfg['estimated_samples']:>12} {cfg['ratio']:>11.1%}")
            total_estimated += cfg['estimated_samples']

        print('-' * 80)
        print(f"{'Total':<15} {'':>6} {'':>14} {total_estimated:>12}")
        print('=' * 80)

        return total_estimated

    def generate_by_global_topn(
        self,
        top_n: int = 20,
        total_samples: int = 1000,
        num_workers: int = 10,
        output_filename: str = "generated_ner_balanced.json"
    ) -> NERSyntheticDataDataset:
        """
        Generate data based on global Top-N entities

        Select top_n entities with highest frequency from all entities, distribute sample count by frequency ratio

        Args:
            top_n: Number of entities to select (sorted by frequency from all entities)
            total_samples: Target total sample count, distributed to each entity by frequency ratio
            num_workers: Number of parallel threads
            output_filename: Output filename

        Returns:
            Generated dataset
        """
        # 1. Get global Top-N entities
        global_top_entities = self.corpus.get_global_top_entities(top_n)

        if not global_top_entities:
            print("Warning: No entities found")
            return NERSyntheticDataDataset()

        # 2. Calculate sample count for each entity (distribute total samples by frequency ratio)
        total_freq = sum(item[2] for item in global_top_entities)
        entity_sample_counts = []

        for entity_name, entity_type, count in global_top_entities:
            # Distribute sample count by frequency ratio, minimum 1
            ratio = count / total_freq
            num_samples = max(1, int(total_samples * ratio))
            entity_sample_counts.append((entity_name, entity_type, count, num_samples))

        total_estimated = sum(item[3] for item in entity_sample_counts)

        print(f"\n{'='*80}")
        print(f"Global Top-{top_n} Entity Generation Configuration (by frequency ratio)")
        print(f"{'='*80}")
        print(f"\n{'Rank':<6} {'Entity Name':<20} {'Type':<12} {'Freq':>8} {'Ratio':>8} {'Samples':>10}")
        print('-' * 80)

        for i, (entity_name, entity_type, count, num_samples) in enumerate(entity_sample_counts, 1):
            ratio = count / total_freq
            print(f"{i:<6} {entity_name:<20} {entity_type:<12} {count:>8} {ratio:>7.1%} {num_samples:>10}")

        print('-' * 80)
        print(f"{'Total':<6} {'':<20} {'':<12} {'':>8} {'':>8} {total_estimated:>10}")
        print('=' * 80)

        print(f"\n{'='*80}")
        print(f"Starting generation (parallelism: {num_workers})")
        print(f"Request strategy: Queue + Stack (failed requests prioritized for retry)")
        print(f"{'='*80}")

        # 3. Create dataset object
        dataset = NERSyntheticDataDataset()

        # 4. Build task queue and retry stack
        task_queue = deque()  # New task queue
        retry_stack = []      # Failed task stack (prioritized)
        max_retries = 10      # Maximum retry count

        for entity_name, entity_type, count, num_samples in entity_sample_counts:
            for _ in range(num_samples):
                task_queue.append((entity_name, entity_type, 0))  # (entity_name, entity_type, retry_count)

        # 5. Generate using queue + stack strategy
        total_generated = 0
        total_failed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            active_tasks = 0

            def submit_task(entity_name, entity_type, retry_count):
                """Submit a single task"""
                nonlocal active_tasks
                future = executor.submit(
                    self.model.generate_single,
                    self.corpus,
                    [entity_name]
                )
                futures[future] = (entity_name, entity_type, retry_count)
                active_tasks += 1

            def get_next_task():
                """Get next task: prioritize stack, then queue"""
                if retry_stack:
                    return retry_stack.pop()
                elif task_queue:
                    return task_queue.popleft()
                return None

            # Initial task filling
            for _ in range(min(num_workers, len(task_queue))):
                task = get_next_task()
                if task:
                    submit_task(*task)

            # Process completed tasks and submit new ones
            while futures:
                # Wait for any task to complete
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # No completed tasks, wait briefly
                    time.sleep(0.1)
                    continue

                for future in done_futures:
                    entity_name, entity_type, retry_count = futures.pop(future)
                    active_tasks -= 1

                    try:
                        sample = future.result()
                        if sample:
                            dataset.add_sample(sample)
                            total_generated += 1
                            print(f"  [{total_generated}/{total_estimated}] OK "
                                  f"[{entity_type}] {entity_name}: {sample.text[:40]}...")
                        else:
                            # Generation failed, add to retry stack
                            if retry_count < max_retries:
                                retry_stack.append((entity_name, entity_type, retry_count + 1))
                                print(f"  X Generation failed (target: {entity_name}), added to retry stack (retry {retry_count + 1}/{max_retries})")
                            else:
                                total_failed += 1
                                print(f"  X Generation failed (target: {entity_name}), max retries reached")
                    except Exception as e:
                        # Exception, add to retry stack
                        if retry_count < max_retries:
                            retry_stack.append((entity_name, entity_type, retry_count + 1))
                            print(f"  X Error: {e}, added to retry stack (retry {retry_count + 1}/{max_retries})")
                        else:
                            total_failed += 1
                            print(f"  X Error: {e}, max retries reached")

                    # Submit new task
                    task = get_next_task()
                    if task:
                        submit_task(*task)

        # Process remaining retry tasks (if any)
        while retry_stack:
            entity_name, entity_type, retry_count = retry_stack.pop()
            if retry_count < max_retries:
                try:
                    sample = self.model.generate_single(self.corpus, [entity_name])
                    if sample:
                        dataset.add_sample(sample)
                        total_generated += 1
                        print(f"  [{total_generated}/{total_estimated}] OK (retry succeeded) "
                              f"[{entity_type}] {entity_name}: {sample.text[:40]}...")
                    else:
                        total_failed += 1
                        print(f"  X Retry failed (target: {entity_name})")
                except Exception as e:
                    total_failed += 1
                    print(f"  X Retry error: {e}")

        # 6. Save dataset
        output_path = self.output_dir / output_filename
        dataset.save(str(output_path))

        # 7. Print statistics
        print(f"\n{'='*80}")
        print(f"Generation completed!")
        print(f"{'='*80}")
        print(f"Target samples: {total_estimated}")
        print(f"Actually generated: {total_generated}")
        print(f"Failed: {total_failed}")
        print(f"Completion rate: {total_generated/total_estimated*100:.1f}%")

        dataset.print_statistics()

        return dataset

    def generate_balanced_data(
        self,
        target_total_samples: int = 2000,
        avg_entities_per_sample: int = 4,
        num_workers: int = 10,
        output_filename: str = "generated_ner_balanced.json"
    ) -> NERSyntheticDataDataset:
        """
        Generate balanced data (distributed by entity type ratio)

        Args:
            target_total_samples: Target total sample count
            avg_entities_per_sample: Average entities per sample
            num_workers: Number of parallel threads
            output_filename: Output filename

        Returns:
            Generated dataset
        """
        # 1. Calculate generation config
        config = self.calculate_generation_config(
            target_total_samples,
            avg_entities_per_sample
        )
        total_estimated = self.print_generation_config(config)

        print(f"\n{'='*80}")
        print(f"Starting balanced data generation (parallelism: {num_workers})")
        print(f"{'='*80}")

        # 2. Create dataset object
        dataset = NERSyntheticDataDataset()

        # 3. Generate by entity type (stratified)
        total_generated = 0

        for entity_type, cfg in sorted(config.items(),
                                         key=lambda x: x[1]['estimated_samples'],
                                         reverse=True):
            if cfg['estimated_samples'] == 0:
                continue

            print(f"\n[{entity_type}] Generating Top-{cfg['top_n']} x "
                  f"{cfg['samples_per_entity']} samples/entity = "
                  f"~{cfg['estimated_samples']} samples...")

            # Get top entities for this type
            top_entities = self.corpus.get_top_entities(entity_type, cfg['top_n'])

            if not top_entities:
                print(f"  Warning: No entities found, skipping")
                continue

            # Parallel generation
            type_samples = 0
            tasks = []

            for entity_name in top_entities:
                for _ in range(cfg['samples_per_entity']):
                    tasks.append((entity_name, entity_type))

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}

                for entity_name, _ in tasks:
                    future = executor.submit(
                        self.model.generate_single,
                        self.corpus,
                        [entity_name]
                    )
                    futures[future] = entity_name

                # Collect results
                for future in as_completed(futures):
                    entity_name = futures[future]

                    try:
                        sample = future.result()
                        if sample:
                            dataset.add_sample(sample)
                            type_samples += 1
                            total_generated += 1
                            entity_count = len(sample.entities)
                            print(f"  [{type_samples}/{cfg['estimated_samples']}] OK "
                                  f"{entity_count} entities: {sample.text[:40]}...")
                        else:
                            print(f"  X Generation failed (target: {entity_name})")
                    except Exception as e:
                        print(f"  X Error: {e}")

            print(f"  Completed {entity_type}: generated {type_samples} samples")

        # 4. Save dataset
        output_path = self.output_dir / output_filename
        dataset.save(str(output_path))

        # 5. Print statistics
        print(f"\n{'='*80}")
        print(f"Generation completed!")
        print(f"{'='*80}")
        print(f"Target samples: {total_estimated}")
        print(f"Actually generated: {total_generated}")
        print(f"Completion rate: {total_generated/total_estimated*100:.1f}%")

        dataset.print_statistics()

        return dataset

    def generate_uniform_multi_entity(
        self,
        total_samples: int = 10000,
        entities_per_sample: int = 3,
        num_workers: int = 10,
        top_n: Optional[int] = None,
        output_filename: str = "generated_ner_uniform_multi.json"
    ) -> NERSyntheticDataDataset:
        """
        Type-level uniform sampling + multi-entity generation

        Strategy:
        1. If top_n is specified, select N entities with highest global frequency (ensure at least one per type)
        2. Each entity type gets equal sample allocation (type-level uniform)
        3. Each sample contains multiple entities (multi-entity generation)
        4. Multiple entities preferably from different types for diversity

        Args:
            total_samples: Target total sample count
            entities_per_sample: Number of entities per sample (default 3, matching test set distribution)
            num_workers: Number of parallel threads
            top_n: Optional, select N entities with highest global frequency (ensure at least one per type)
            output_filename: Output filename

        Returns:
            Generated dataset
        """
        import random
        from itertools import cycle

        # 1. Get entities: if top_n specified, use global Top-N strategy
        if top_n is not None:
            # Use global Top-N to select entities (ensure at least one per type)
            top_entities = self.corpus.get_global_top_entities(top_n)

            # Group by type
            type_entities = {}
            for entity_name, entity_type, count in top_entities:
                if entity_type not in type_entities:
                    type_entities[entity_type] = []
                type_entities[entity_type].append(entity_name)

            entity_types = list(type_entities.keys())

            print(f"\n{'='*80}")
            print(f"Top-N Type-level Uniform Sampling + Multi-entity Generation Configuration")
            print(f"{'='*80}")
            print(f"Top-N: {top_n}")
            print(f"Entity types covered: {len(entity_types)}")
            print(f"Target total samples: {total_samples}")
            print(f"Entities per sample: {entities_per_sample}")
            print(f"Parallel threads: {num_workers}")
            print(f"\nSelected Top-{top_n} entities:")
            for entity_name, entity_type, count in top_entities:
                print(f"  - {entity_name} [{entity_type}]: freq {count}")
            print(f"\nDistribution by type:")
            for entity_type in entity_types:
                entities = type_entities[entity_type]
                print(f"  - {entity_type}: {len(entities)} entities")
        else:
            # Use all entities from all entity types
            entity_types = list(self.corpus.entity_distribution.keys())
            # Filter out types with no entities
            entity_types = [t for t in entity_types
                           if self.corpus.entity_distribution[t]['count'] > 0]

            # Collect entities for each type
            type_entities = {}
            for entity_type in entity_types:
                entities = self.corpus.get_top_entities(entity_type, n=1000)
                type_entities[entity_type] = entities if entities else []

            print(f"\n{'='*80}")
            print(f"Type-level Uniform Sampling + Multi-entity Generation Configuration")
            print(f"{'='*80}")
            print(f"Entity type count: {len(entity_types)}")
            print(f"Target total samples: {total_samples}")
            print(f"Entities per sample: {entities_per_sample}")
            print(f"Parallel threads: {num_workers}")
            print(f"\nEntity type distribution:")
            for entity_type in entity_types:
                count = len(type_entities.get(entity_type, []))
                print(f"  - {entity_type}: {count} entities")

        num_types = len(entity_types)
        samples_per_type = total_samples // num_types

        print(f"\nSamples per type: {samples_per_type}")
        print(f"{'='*80}")

        # 3. Create task list: each task contains multiple entities (from different types)
        tasks = []
        type_cycle = cycle(entity_types)
        type_sample_counts = {t: 0 for t in entity_types}

        for _ in range(total_samples):
            # Select primary type (rotation ensures uniformity)
            primary_type = next(type_cycle)

            # Find a type that hasn't reached its quota (with max attempts to avoid infinite loop)
            max_attempts = len(entity_types) * 2
            attempts = 0
            while type_sample_counts[primary_type] >= samples_per_type and attempts < max_attempts:
                primary_type = next(type_cycle)
                attempts += 1

            # If all types are full, pick the one with minimum samples
            if attempts >= max_attempts:
                primary_type = min(type_sample_counts, key=type_sample_counts.get)

            type_sample_counts[primary_type] += 1

            # Select primary entity
            primary_entities = type_entities.get(primary_type, [])
            if not primary_entities:
                continue
            primary_entity = random.choice(primary_entities)

            # Select additional entities (from other types)
            selected_entities = [(primary_entity, primary_type)]
            other_types = [t for t in entity_types if t != primary_type]

            for _ in range(entities_per_sample - 1):
                if not other_types:
                    break
                other_type = random.choice(other_types)
                other_entities = type_entities.get(other_type, [])
                if other_entities:
                    other_entity = random.choice(other_entities)
                    selected_entities.append((other_entity, other_type))
                    other_types.remove(other_type)

            tasks.append(selected_entities)

        print(f"\nTask generation completed: {len(tasks)} tasks")
        print(f"Actual allocation per type:")
        for t, c in sorted(type_sample_counts.items(), key=lambda x: -x[1]):
            print(f"  - {t}: {c} samples")

        # 4. Parallel generation
        print(f"\n{'='*80}")
        print(f"Starting generation (parallelism: {num_workers})")
        print(f"{'='*80}")

        dataset = NERSyntheticDataDataset()
        total_generated = 0
        total_failed = 0

        # Queue + Stack strategy
        task_queue = list(tasks)
        retry_stack = []
        max_retries = 10

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}

            # Initial submission
            initial_batch = min(num_workers * 2, len(task_queue))
            for _ in range(initial_batch):
                if task_queue:
                    task = task_queue.pop(0)
                    entity_names = [e[0] for e in task]
                    future = executor.submit(
                        self.model.generate_single,
                        self.corpus,
                        entity_names
                    )
                    futures[future] = (task, 0)  # (task, retry_count)

            while futures:
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    import time
                    time.sleep(0.1)
                    continue

                for future in done_futures:
                    task, retry_count = futures.pop(future)
                    entity_names = [e[0] for e in task]
                    entity_types_str = ", ".join([e[1] for e in task])

                    try:
                        sample = future.result()
                        if sample:
                            dataset.add_sample(sample)
                            total_generated += 1
                            entity_count = len(sample.entities)
                            print(f"  [{total_generated}/{total_samples}] OK "
                                  f"{entity_count} entities [{entity_types_str}]: "
                                  f"{sample.text[:40]}...")
                        else:
                            if retry_count < max_retries:
                                retry_stack.append((task, retry_count + 1))
                                print(f"  X Generation failed, added to retry queue "
                                      f"(target: {', '.join(entity_names)})")
                            else:
                                total_failed += 1
                                print(f"  X Generation failed, max retries reached")
                    except Exception as e:
                        if retry_count < max_retries:
                            retry_stack.append((task, retry_count + 1))
                            print(f"  X Error: {e}, added to retry queue")
                        else:
                            total_failed += 1
                            print(f"  X Error: {e}, max retries reached")

                    # Submit new task: prioritize retry stack
                    if retry_stack:
                        task, retry_count = retry_stack.pop()
                        entity_names = [e[0] for e in task]
                        future = executor.submit(
                            self.model.generate_single,
                            self.corpus,
                            entity_names
                        )
                        futures[future] = (task, retry_count)
                    elif task_queue:
                        task = task_queue.pop(0)
                        entity_names = [e[0] for e in task]
                        future = executor.submit(
                            self.model.generate_single,
                            self.corpus,
                            entity_names
                        )
                        futures[future] = (task, 0)

        # Process remaining retry tasks
        while retry_stack:
            task, retry_count = retry_stack.pop()
            if retry_count < max_retries:
                try:
                    entity_names = [e[0] for e in task]
                    sample = self.model.generate_single(self.corpus, entity_names)
                    if sample:
                        dataset.add_sample(sample)
                        total_generated += 1
                        print(f"  [{total_generated}/{total_samples}] OK (retry succeeded)")
                    else:
                        total_failed += 1
                except Exception as e:
                    total_failed += 1

        # 5. Save dataset
        output_path = self.output_dir / output_filename
        dataset.save(str(output_path))

        # 6. Print statistics
        print(f"\n{'='*80}")
        print(f"Generation completed!")
        print(f"{'='*80}")
        print(f"Target samples: {total_samples}")
        print(f"Actually generated: {total_generated}")
        print(f"Failed: {total_failed}")
        print(f"Completion rate: {total_generated/total_samples*100:.1f}%")

        dataset.print_statistics()

        return dataset

    def verify_distribution(
        self,
        generated_dataset: NERSyntheticDataDataset
    ) -> Dict[str, float]:
        """
        Verify distribution difference between generated data and real data

        Args:
            generated_dataset: Generated dataset

        Returns:
            {entity_type: diff_ratio}
        """
        generated_dist = generated_dataset.analyze_distribution()
        generated_total = sum(generated_dist.values())

        print(f"\n{'='*80}")
        print(f"Distribution Comparison Verification")
        print(f"{'='*80}")
        print(f"\n{'Entity Type':<15} {'Real Ratio':>10} {'Gen Ratio':>10} {'Diff':>10}")
        print('-' * 80)

        diffs = {}

        for entity_type in sorted(self.corpus.entity_distribution.keys()):
            real_ratio = self.corpus.entity_distribution[entity_type]['ratio']
            gen_count = generated_dist.get(entity_type, 0)
            gen_ratio = gen_count / generated_total if generated_total > 0 else 0
            diff = gen_ratio - real_ratio

            diffs[entity_type] = diff

            diff_str = f"{diff:+.1%}" if diff != 0 else "0.0%"
            print(f"{entity_type:<15} {real_ratio:>9.1%} {gen_ratio:>9.1%} {diff_str:>10}")

        print('=' * 80)

        # Calculate overall deviation
        avg_diff = sum(abs(d) for d in diffs.values()) / len(diffs)
        print(f"\nMean absolute deviation: {avg_diff:.1%}")

        if avg_diff < 0.02:
            print("Excellent: Distribution very close to real data")
        elif avg_diff < 0.05:
            print("Good: Distribution reasonably close to real data")
        else:
            print("Warning: Distribution deviation is large, may need adjustment")

        return diffs


# Convenience functions

def create_balanced_dataset(
    real_dataset_path: str,
    output_dir: str = "dataset/yakusha_synthetic_balanced",
    target_samples: int = 2000,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    num_workers: int = 10
) -> NERSyntheticDataDataset:
    """
    Convenience function: One-click balanced dataset generation

    Args:
        real_dataset_path: Path to real dataset
        output_dir: Output directory
        target_samples: Target sample count
        api_key: OpenAI API key
        model: Model name
        num_workers: Number of parallel threads

    Returns:
        Generated dataset
    """
    # 1. Load corpus
    corpus = NERSyntheticDataCorpus(real_dataset_path)
    corpus.load()
    corpus.print_statistics()

    # 2. Initialize model
    gen_model = NERSyntheticDataModel(
        api_key=api_key,
        model=model
    )

    # 3. Create generator
    generator = NERSyntheticDataGenerator(
        corpus=corpus,
        model=gen_model,
        output_dir=output_dir
    )

    # 4. Generate data
    dataset = generator.generate_balanced_data(
        target_total_samples=target_samples,
        num_workers=num_workers
    )

    # 5. Verify distribution
    generator.verify_distribution(dataset)

    return dataset


def create_uniform_multi_dataset(
    real_dataset_path: str,
    output_dir: str = "dataset/yakusha_synthetic_uniform_multi",
    total_samples: int = 10000,
    entities_per_sample: int = 3,
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    num_workers: int = 10,
    provider: str = "openai"
) -> NERSyntheticDataDataset:
    """
    Convenience function: Type-level uniform sampling + multi-entity generation

    Strategy:
    - If top_n is specified, select N entities with highest global frequency (ensure at least one per type)
    - Each entity type gets equal sample allocation
    - Each sample contains multiple entities (from different types)
    - Suitable for improving low-frequency entity recognition accuracy

    Args:
        real_dataset_path: Path to real dataset
        output_dir: Output directory
        total_samples: Target total sample count
        entities_per_sample: Number of entities per sample (default 3, matching test set distribution)
        top_n: Optional, select N entities with highest global frequency
        api_key: API key
        model: Model name
        num_workers: Number of parallel threads
        provider: API provider ("openai" or "claude")

    Returns:
        Generated dataset
    """
    # 1. Load corpus
    corpus = NERSyntheticDataCorpus(real_dataset_path)
    corpus.load()
    corpus.print_statistics()

    # 2. Initialize model
    gen_model = NERSyntheticDataModel(
        api_key=api_key,
        model=model,
        provider=provider
    )

    # 3. Create generator
    generator = NERSyntheticDataGenerator(
        corpus=corpus,
        model=gen_model,
        output_dir=output_dir
    )

    # 4. Generate data
    dataset = generator.generate_uniform_multi_entity(
        total_samples=total_samples,
        entities_per_sample=entities_per_sample,
        num_workers=num_workers,
        top_n=top_n
    )

    # 5. Verify distribution (shows difference between uniform and original distribution)
    generator.verify_distribution(dataset)

    return dataset


def create_topn_dataset(
    real_dataset_path: str,
    output_dir: str = "dataset/yakusha_synthetic_topn",
    top_n: int = 20,
    total_samples: int = 1000,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    num_workers: int = 10,
    provider: str = "openai"
) -> NERSyntheticDataDataset:
    """
    Convenience function: Generate dataset based on global Top-N entities

    Select top_n entities with highest frequency from all entities, distribute total samples by frequency ratio

    Args:
        real_dataset_path: Path to real dataset
        output_dir: Output directory
        top_n: Number of entities to select (sorted by frequency from all entities)
        total_samples: Target total sample count, distributed to each entity by frequency ratio
        api_key: API key (OpenAI or Anthropic)
        model: Model name
        num_workers: Number of parallel threads
        provider: API provider ("openai" or "claude")

    Returns:
        Generated dataset
    """
    # 1. Load corpus
    corpus = NERSyntheticDataCorpus(real_dataset_path)
    corpus.load()
    corpus.print_statistics()

    # 2. Initialize model
    gen_model = NERSyntheticDataModel(
        api_key=api_key,
        model=model,
        provider=provider
    )

    # 3. Create generator
    generator = NERSyntheticDataGenerator(
        corpus=corpus,
        model=gen_model,
        output_dir=output_dir
    )

    # 4. Generate data
    dataset = generator.generate_by_global_topn(
        top_n=top_n,
        total_samples=total_samples,
        num_workers=num_workers
    )

    # 5. Verify distribution
    generator.verify_distribution(dataset)

    return dataset

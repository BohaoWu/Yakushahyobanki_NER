#!/usr/bin/env python3
"""
Core classes for synthetic NER data generation

Class structure:
- NERSyntheticDataCorpus: Corpus containing real data for analysis and few-shot examples
- NERSyntheticDataModel: Generation model configuration, including LLM settings and prompt templates
- NERSyntheticDataDataset: Dataset class representing generated or loaded NER data
- NERSyntheticDataGenerator: Generator that orchestrates the entire generation pipeline
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datasets import load_dataset, Dataset, DatasetDict
import random

@dataclass
class EntityExample:
    """Entity example"""
    name: str
    type: str
    span: Tuple[int, int]


@dataclass
class NERSample:
    """Single NER sample"""
    text: str
    entities: List[Dict[str, any]]

    def to_dict(self):
        return {
            'text': self.text,
            'entities': self.entities
        }


class NERSyntheticDataCorpus:
    """
    Corpus class

    Features:
    1. Load real datasets
    2. Analyze entity distribution
    3. Extract entity examples
    4. Provide few-shot examples
    """

    def __init__(self, dataset_path: str):
        """
        Initialize corpus

        Args:
            dataset_path: Path to the real dataset
        """
        self.dataset_path = Path(dataset_path)
        self.samples = []
        self.entity_types = set()
        self.entity_distribution = {}
        self.entity_examples = defaultdict(list)
        self.total_entities = 0

    def load(self, trust_remote_code: bool = True, exclude_test: bool = True):
        """Load dataset

        Args:
            trust_remote_code: Whether to trust remote code
            exclude_test: Whether to exclude test set to avoid data leakage (default: True)
        """
        print(f"Loading corpus: {self.dataset_path}")

        try:
            dataset = load_dataset(str(self.dataset_path), trust_remote_code=trust_remote_code)

            # Only use train and validation for entity statistics to avoid data leakage
            splits_to_use = ['train', 'validation'] if exclude_test else ['train', 'validation', 'test']
            print(f"  Using splits: {splits_to_use}" + (" (excluding test set to avoid leakage)" if exclude_test else ""))

            for split in splits_to_use:
                if split in dataset:
                    for example in dataset[split]:
                        self._process_example(example)

            self._analyze_distribution()
            print(f"Loading complete: {len(self.samples)} samples, {self.total_entities} entities")

        except Exception as e:
            print(f"Error: Loading failed - {e}")
            raise

    def _process_example(self, example: Dict):
        """Process a single sample"""
        text = example.get('text', '')
        entities = example.get('entities', [])

        if not entities:
            return

        sample = NERSample(
            text=text,
            entities=[]
        )

        for entity in entities:
            entity_type = entity.get('type', '')
            entity_name = entity.get('name', '')
            span = entity.get('span', [0, 0])

            self.entity_types.add(entity_type)

            # Filter out garbled characters
            if not any(c in entity_name for c in ['?', '☆', '〈', '〉']):
                self.entity_examples[entity_type].append(entity_name)

            sample.entities.append({
                'type': entity_type,
                'name': entity_name,
                'span': span
            })

        self.samples.append(sample)

    def _analyze_distribution(self):
        """Analyze entity distribution"""
        entity_counts = Counter()

        for sample in self.samples:
            for entity in sample.entities:
                entity_counts[entity['type']] += 1

        self.total_entities = sum(entity_counts.values())

        for entity_type, count in entity_counts.items():
            unique_entities = len(set(self.entity_examples[entity_type]))
            self.entity_distribution[entity_type] = {
                'count': count,
                'ratio': count / self.total_entities,
                'unique_entities': unique_entities
            }

    def get_few_shot_examples(self, n: int = 3, min_entities: int = 10) -> List[NERSample]:
        """
        Get few-shot examples

        Args:
            n: Number of examples to return
            min_entities: Minimum number of entities to include

        Returns:
            List of examples
        """
        # Filter out samples containing garbled characters
        bad_chars = ['〈', '〉', '☆', '?']
        clean_samples = []

        for sample in self.samples:
            if any(c in sample.text for c in bad_chars):
                continue

            has_bad_entity = False
            for entity in sample.entities:
                if any(c in entity['name'] for c in bad_chars):
                    has_bad_entity = True
                    break

            if not has_bad_entity and len(sample.entities) >= min_entities:
                clean_samples.append(sample)

        # Sort by entity count, return top n
        clean_samples.sort(key=lambda s: len(s.entities), reverse=True)
        return clean_samples[:n]

    def get_top_entities(self, entity_type: str, n: int) -> List[str]:
        """
        Get Top-N most frequent entities of a given type

        Args:
            entity_type: Entity type
            n: Number to return

        Returns:
            List of entity names
        """
        if entity_type not in self.entity_examples:
            return []

        entity_freq = Counter(self.entity_examples[entity_type])
        return [name for name, _ in entity_freq.most_common(n)]

    def get_global_top_entities(self, n: int) -> List[Tuple[str, str, int]]:
        """
        Get Top-N most frequent entities across all entity types

        Selection strategy:
        1. First ensure each entity type has at least one entity selected (the most frequent one of that type)
        2. Fill remaining slots by global frequency ranking

        Args:
            n: Number to return

        Returns:
            [(entity_name, entity_type, count), ...] list, sorted by frequency in descending order
        """
        # Count frequencies of all entities (including type information)
        all_entity_freq = Counter()

        for entity_type, entities in self.entity_examples.items():
            for entity_name in entities:
                key = (entity_name, entity_type)
                all_entity_freq[key] += 1

        # Step 1: Select the most frequent entity for each entity type
        selected = set()  # Selected (entity_name, entity_type) pairs
        result = []

        for entity_type in self.entity_types:
            # Get all entities of this type sorted by frequency
            type_entities = [
                ((name, etype), count)
                for (name, etype), count in all_entity_freq.items()
                if etype == entity_type
            ]
            if type_entities:
                type_entities.sort(key=lambda x: x[1], reverse=True)
                top_entity, count = type_entities[0]
                if top_entity not in selected:
                    selected.add(top_entity)
                    result.append((top_entity[0], top_entity[1], count))

        # Step 2: If there are remaining slots, fill them by global frequency ranking
        remaining_slots = n - len(result)
        if remaining_slots > 0:
            # Get all unselected entities, sorted by frequency
            remaining_entities = [
                ((name, etype), count)
                for (name, etype), count in all_entity_freq.items()
                if (name, etype) not in selected
            ]
            remaining_entities.sort(key=lambda x: x[1], reverse=True)

            for (name, etype), count in remaining_entities[:remaining_slots]:
                result.append((name, etype, count))

        # Re-sort final results by frequency
        result.sort(key=lambda x: x[2], reverse=True)

        return result[:n]

    def print_statistics(self):
        """Print statistics"""
        print(f"\n{'='*80}")
        print(f"Corpus Statistics")
        print(f"{'='*80}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Total entities: {self.total_entities}")
        print(f"Entity type count: {len(self.entity_types)}")
        print(f"\n{'Entity Type':<15} {'Count':>8} {'Ratio':>10} {'Unique Entities':>15}")
        print('-' * 80)

        for entity_type, stats in sorted(self.entity_distribution.items(),
                                          key=lambda x: x[1]['count'], reverse=True):
            print(f"{entity_type:<15} {stats['count']:>8} "
                  f"{stats['ratio']:>9.1%} {stats['unique_entities']:>15}")

        print('=' * 80)


class NERSyntheticDataModel:
    """
    Generation model class

    Features:
    1. Configure LLM parameters (model, api_key, temperature, etc.)
    2. Generate prompt templates
    3. Call LLM API to generate data
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        provider: str = "openai"
    ):
        """
        Initialize generation model

        Args:
            api_key: API key
            model: Model name
            temperature: Temperature parameter
            base_url: API base URL (OpenAI only)
            provider: API provider ("openai" or "claude")
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.provider = provider.lower()
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize API client"""
        if self.provider == "claude":
            self._initialize_claude_client()
        else:
            self._initialize_openai_client()

    def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            # Try to get from config or environment variables
            try:
                import sys
                from pathlib import Path
                PROJECT_ROOT = Path(__file__).parent.parent.parent
                sys.path.insert(0, str(PROJECT_ROOT))
                from src.config.config import get_openai_api_key, OPENAI_BASE_URL
                self.api_key = get_openai_api_key()
                if not self.base_url and OPENAI_BASE_URL:
                    self.base_url = OPENAI_BASE_URL
            except:
                pass

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None
                )
                print(f"OpenAI client initialized (model: {self.model})")
            except ImportError:
                print("Warning: openai library not installed")
        else:
            print("Warning: OpenAI API key not found")

    def _initialize_claude_client(self):
        """Initialize Claude client"""
        if not self.api_key:
            # Try to get from environment variables
            import os
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not self.api_key:
                try:
                    import sys
                    from pathlib import Path
                    PROJECT_ROOT = Path(__file__).parent.parent.parent
                    sys.path.insert(0, str(PROJECT_ROOT))
                    from src.config.config import get_anthropic_api_key
                    self.api_key = get_anthropic_api_key()
                except:
                    pass

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print(f"Anthropic client initialized (model: {self.model})")
            except ImportError:
                print("Warning: anthropic library not installed, please run: pip install anthropic")
        else:
            print("Warning: ANTHROPIC_API_KEY not found")

    def create_prompt(
        self,
        corpus: NERSyntheticDataCorpus,
        target_entities: Optional[List[str]] = None
    ) -> str:
        """
        Create generation prompt

        Args:
            corpus: Corpus
            target_entities: Target entities (entities that must be included)

        Returns:
            Prompt string
        """
        # Entity type definitions (only types present in the corpus)
        entity_definitions = {
            "役者": "[Most Important] Kabuki actor names. The primary entity comprising over 60% of the data",
            "興行関係者": "[Important] Theater managers, promoters, etc. Approximately 18% of the data",
            "俳名": "[Important] Actor pen names (haigo). Approximately 8% of the data",
            "演目名": "[Somewhat Important] Kabuki play titles. Approximately 6% of the data",
            "人名": "[Normal] Person names other than actors. Approximately 3% of the data",
            "書名": "[Somewhat Rare] Kabuki-related books and literary works",
            "狂言作者": "[Somewhat Rare] Playwrights, dramatists",
            "役名": "[Rare] Character names in plays",
            "屋号": "[Rare] Kabuki actor house names (yago)",
            "音曲": "[Rare] Musicians, joruri performers, etc.",
            "事項": "[Extremely Rare] Important matters. Rarely used",
        }

        # Only include entity types that exist in the corpus
        entity_definitions_str = []
        for entity_type in sorted(corpus.entity_types):
            definition = entity_definitions.get(entity_type, "Related entity")
            examples = corpus.get_top_entities(entity_type, 5)
            examples_str = ", ".join(examples) if examples else "(no examples)"
            entity_definitions_str.append(
                f"- {entity_type} ({definition})\n  Examples: {examples_str}"
            )

        # Few-shot examples
        few_shot_samples = corpus.get_few_shot_examples(n=3)
        few_shot_examples = []

        for i, sample in enumerate(few_shot_samples, 1):
            entities_list = []
            for entity in sample.entities:
                entities_list.append(
                    f"    - [{entity['type']}] {entity['name']} "
                    f"(position: {entity['span'][0]}-{entity['span'][1]})"
                )

            few_shot_examples.append(
                f"""Example {i}:
Generated text: {sample.text}
Annotation:
{chr(10).join(entities_list)}"""
            )

        # Task description
        task_description = "Generate one passage related to Edo-period kabuki and Yakushahyoubanki, and identify and annotate the named entities (NER) in the passage."
        if target_entities:
            entities_str = ", ".join(target_entities)
            task_description += f"\n\nImportant: Please make sure to include the following entities in the passage: {entities_str}"

        # Complete prompt
        prompt = f"""# Domain Specification
You are an expert well-versed in the fields of Japanese Edo-period Yakushahyoubanki and kabuki. You can perfectly reproduce the writing style of Edo-period theater critiques, actor reviews, and theater-related literature (especially classical literary style from the Genroku and Hoei periods).

# Task
{task_description}

# Entity Types
Please generate a passage containing the following types of entities:

{chr(10).join(entity_definitions_str)}

# Generation Guidelines
**Important: Generate in Edo-period classical literary style (original style of the Genroku and Hoei periods)**

Writing style characteristics:
1. Use historical kana orthography (e.g., "wi", "we", "keri", "haberi")
2. Use classical grammar (e.g., "nari", "keri", "haberu", "beshi", "ran")
3. Sensibility of variant kana and old character forms
4. Include multiple contents in a single long sentence (approximately 50-120 characters)
5. Actor review and theater critique style (ending with "~nari", "~keri", "~haberu")
6. Include metaphorical and erudite expressions

Characteristics of example passages:
- "~nite enjirare-shi", "~no yaku wo tsutome", "migoto nari"
- Conjunctions such as "kono tabi", "satemo", "saredo", "koko ni"
- Auxiliary verbs such as "haberu", "soro", "nari", "keri"
- Expressions such as "~to oboeyu", "~to iu"

# Few-shot Examples
The following are actual examples from Edo-period literature:

{chr(10).join(few_shot_examples)}

# Output Format
Please output in the following JSON format:

<start_annotation>
{{
    "text": "Generated Edo-period style passage",
    "entities": [
        {{
            "type": "entity type",
            "name": "entity text",
            "start": start position (integer),
            "end": end position (integer)
        }}
    ]
}}
</end_annotation>

Notes:
- In "text", write the generated Edo-period style passage
- start/end are character indices (starting from 0)
- Calculate entity positions accurately

Now, please generate a new passage:"""

        return prompt

    def generate_single(
        self,
        corpus: NERSyntheticDataCorpus,
        target_entities: Optional[List[str]] = None
    ) -> Optional[NERSample]:
        """
        Generate a single sample

        Args:
            corpus: Corpus
            target_entities: Target entities

        Returns:
            Generated sample or None
        """
        if not self.client:
            print(f"Error: {self.provider} client not initialized")
            return None

        try:
            prompt = self.create_prompt(corpus, target_entities)

            if self.provider == "claude":
                result_text = self._generate_with_claude(prompt)
            else:
                result_text = self._generate_with_openai(prompt)

            if result_text:
                result_dict = self._extract_json(result_text)
                if result_dict:
                    return NERSample(
                        text=result_dict['text'],
                        entities=result_dict['entities']
                    )

            return None

        except Exception as e:
            print(f"Generation failed: {e}")
            return None

    def _generate_with_openai(self, prompt: str) -> Optional[str]:
        """Generate using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in Edo-period kabuki and Yakushahyoubanki literature."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def _generate_with_claude(self, prompt: str) -> Optional[str]:
        """Generate using Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system="You are an expert in Edo-period kabuki and Yakushahyoubanki literature.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    def batch_generate_with_claude(
        self,
        corpus: NERSyntheticDataCorpus,
        target_entities_list: List[List[str]],
        poll_interval: int = 15,
        chunk_size: int = 10,
    ) -> List[Optional['NERSample']]:
        """
        Generate multiple samples via Claude Message Batches API, with chunking.

        Submits requests in parallel chunks (default 100 per chunk) to leverage
        Anthropic's batch parallelism while keeping per-batch overhead amortized.

        Args:
            corpus: Corpus for prompt construction
            target_entities_list: List of target entity lists (one per sample)
            poll_interval: Seconds between status polls (default 15)
            chunk_size: Requests per batch chunk (default 100)

        Returns:
            List of NERSample (or None for failed requests), indexed by request order
        """
        if self.provider != "claude" or self.client is None:
            raise RuntimeError("batch_generate_with_claude requires initialized Claude client")

        import time
        from anthropic.types.messages.batch_create_params import Request

        system_msg = self.SYSTEM_MESSAGES.get(self.lang, self.SYSTEM_MESSAGES["en"])
        total = len(target_entities_list)

        # ----- Submit all chunks in PARALLEL (non-blocking) -----
        chunk_metadata = []  # list of (batch_id, start_idx, end_idx)
        n_chunks = (total + chunk_size - 1) // chunk_size

        print(f"[Batch] Submitting {total} requests in {n_chunks} chunks of {chunk_size}...", flush=True)
        for chunk_idx in range(n_chunks):
            start_i = chunk_idx * chunk_size
            end_i = min(start_i + chunk_size, total)
            chunk = target_entities_list[start_i:end_i]

            requests = []
            for local_i, target_entities in enumerate(chunk):
                global_i = start_i + local_i
                prompt = self.create_prompt(corpus, target_entities)
                requests.append(Request(
                    custom_id=f"req_{global_i:06d}",
                    params={
                        "model": self.model,
                        "max_tokens": 1000,
                        "system": system_msg,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                    }
                ))
            batch = self.client.messages.batches.create(requests=requests)
            chunk_metadata.append((batch.id, start_i, end_i))
            print(f"[Batch]   chunk {chunk_idx+1}/{n_chunks}: {batch.id} ({len(requests)} req)", flush=True)

        # ----- Poll all chunks concurrently -----
        pending_ids = {meta[0] for meta in chunk_metadata}
        ended_ids = set()
        start = time.time()

        while pending_ids:
            time.sleep(poll_interval)
            elapsed = int(time.time() - start)
            for batch_id in list(pending_ids):
                try:
                    b = self.client.messages.batches.retrieve(batch_id)
                    if b.processing_status == "ended":
                        pending_ids.discard(batch_id)
                        ended_ids.add(batch_id)
                except Exception as e:
                    print(f"[Batch] Poll error {batch_id}: {e}", flush=True)
            print(f"[Batch] {len(ended_ids)}/{len(chunk_metadata)} chunks ended (elapsed={elapsed}s)", flush=True)

        print(f"[Batch] All chunks ended. Retrieving results...", flush=True)

        # Build result map across all chunks
        results_map: Dict[int, Optional[NERSample]] = {}
        for batch_id, _, _ in chunk_metadata:
            for result in self.client.messages.batches.results(batch_id):
                idx = int(result.custom_id.split("_")[1])
                if result.result.type == "succeeded":
                    message = result.result.message
                    try:
                        text_content = message.content[0].text
                        result_dict = self._extract_json(text_content)
                        if result_dict:
                            results_map[idx] = NERSample(
                                text=result_dict["text"],
                                entities=result_dict["entities"],
                            )
                        else:
                            results_map[idx] = None
                    except Exception as e:
                        print(f"[Batch] Parse error for {result.custom_id}: {e}", flush=True)
                        results_map[idx] = None
                else:
                    results_map[idx] = None

        # Return in order
        samples = [results_map.get(i) for i in range(total)]
        success = sum(1 for s in samples if s is not None)
        print(f"[Batch] Parsed {success}/{total} successful samples", flush=True)
        return samples

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text"""
        import re

        # Search for <start_annotation> tag
        match = re.search(
            r'<start_annotation>\s*(.*?)\s*</end_annotation>',
            text,
            re.DOTALL
        )
        if match:
            text = match.group(1)

        # Search for JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if 'text' in result and 'entities' in result:
                    return self._validate_and_fix_positions(result)
            except json.JSONDecodeError:
                pass

        return None

    def _validate_and_fix_positions(self, sample: Dict) -> Dict:
        """Validate and fix entity positions"""
        text = sample.get('text', '')
        entities = sample.get('entities', [])

        fixed_entities = []
        for entity in entities:
            entity_name = entity.get('name', '')
            start = entity.get('start', -1)
            end = entity.get('end', -1)
            entity_type = entity.get('type', '')

            # Check if position is correct
            if 0 <= start < end <= len(text) and text[start:end] == entity_name:
                fixed_entities.append({
                    'name': entity_name,
                    'span': [start, end],
                    'type': entity_type
                })
            else:
                # Try to find in text
                found_pos = text.find(entity_name)
                if found_pos >= 0:
                    fixed_entities.append({
                        'name': entity_name,
                        'span': [found_pos, found_pos + len(entity_name)],
                        'type': entity_type
                    })

        sample['entities'] = fixed_entities
        return sample


class NERSyntheticDataDataset:
    """
    Dataset class

    Features:
    1. Store generated samples
    2. Save/load data
    3. Statistical analysis
    4. Convert to Hugging Face datasets format
    """

    def __init__(self, samples: Optional[List[NERSample]] = None):
        """
        Initialize dataset

        Args:
            samples: List of samples
        """
        self.samples = samples or []

    def add_sample(self, sample: NERSample):
        """Add a sample"""
        self.samples.append(sample)

    def save(self, output_path: str):
        """
        Save dataset

        Args:
            output_path: Output path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [sample.to_dict() for sample in self.samples]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Dataset saved: {output_file} ({len(self.samples)} samples)")

    @classmethod
    def load(cls, input_path: str) -> 'NERSyntheticDataDataset':
        """
        Load dataset

        Args:
            input_path: Input path

        Returns:
            Dataset object
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = [
            NERSample(text=item['text'], entities=item['entities'])
            for item in data
        ]

        return cls(samples)

    def analyze_distribution(self) -> Dict[str, int]:
        """Analyze entity distribution"""
        entity_counts = Counter()

        for sample in self.samples:
            for entity in sample.entities:
                entity_counts[entity['type']] += 1

        return dict(entity_counts)

    def print_statistics(self):
        """Print statistics"""
        distribution = self.analyze_distribution()
        total = sum(distribution.values())

        print(f"\n{'='*80}")
        print(f"Dataset Statistics")
        print(f"{'='*80}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Total entities: {total}")
        print(f"\n{'Entity Type':<15} {'Count':>8} {'Ratio':>10}")
        print('-' * 80)

        for entity_type, count in sorted(distribution.items(),
                                          key=lambda x: x[1], reverse=True):
            ratio = count / total if total > 0 else 0
            print(f"{entity_type:<15} {count:>8} {ratio:>9.1%}")

        print('=' * 80)

    def to_huggingface_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> DatasetDict:
        """
        Convert to Hugging Face datasets format

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            DatasetDict
        """
        # Random shuffle
        samples = self.samples.copy()
        random.shuffle(samples)

        # Split dataset
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_data = [s.to_dict() for s in samples[:train_size]]
        val_data = [s.to_dict() for s in samples[train_size:train_size+val_size]]
        test_data = [s.to_dict() for s in samples[train_size+val_size:]]

        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })


# Continue to next file...

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
        """Load dataset (supports HuggingFace datasets and HIPE TSV format)

        Args:
            trust_remote_code: Whether to trust remote code
            exclude_test: Whether to exclude test set to avoid data leakage (default: True)
        """
        import glob as glob_module
        import os

        print(f"Loading corpus: {self.dataset_path}")

        # Check for HIPE TSV format first
        tsv_files = glob_module.glob(os.path.join(str(self.dataset_path), '**/*.tsv'), recursive=True)
        if tsv_files:
            self._load_tsv(tsv_files)
        else:
            try:
                dataset = load_dataset(str(self.dataset_path), trust_remote_code=trust_remote_code)

                # Only use train and validation for entity statistics to avoid data leakage
                splits_to_use = ['train', 'validation'] if exclude_test else ['train', 'validation', 'test']
                print(f"  Using splits: {splits_to_use}" + (" (excluding test set to avoid leakage)" if exclude_test else ""))

                for split in splits_to_use:
                    if split in dataset:
                        for example in dataset[split]:
                            self._process_example(example)
            except Exception as e:
                print(f"Error: Loading failed - {e}")
                raise

        self._analyze_distribution()
        print(f"Loading complete: {len(self.samples)} samples, {self.total_entities} entities")

    def _load_tsv(self, tsv_files):
        """Load HIPE TSV format files into corpus samples"""
        for filepath in tsv_files:
            sentences = []
            current_sentence = []

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line:
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_sentence.append((parts[0], parts[1]))
                    elif len(parts) == 1:
                        current_sentence.append((parts[0], 'O'))
                if current_sentence:
                    sentences.append(current_sentence)

            for sentence in sentences:
                tokens = [t[0] for t in sentence]
                tags = [t[1] for t in sentence]
                text = ' '.join(tokens)

                entities = []
                char_pos = 0
                current_entity = None

                for token, tag in zip(tokens, tags):
                    token_start = char_pos
                    token_end = char_pos + len(token)

                    if tag.startswith('B-'):
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            'name': token,
                            'span': [token_start, token_end],
                            'type': tag[2:],
                        }
                    elif tag.startswith('I-') and current_entity:
                        current_entity['name'] += ' ' + token
                        current_entity['span'][1] = token_end
                    else:
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None

                    char_pos = token_end + 1

                if current_entity:
                    entities.append(current_entity)

                if entities:
                    self._process_example({'text': text, 'entities': entities})

        print(f"  Loaded {len(tsv_files)} TSV file(s)")

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

    # Multilingual entity definitions
    ENTITY_DEFINITIONS_BY_LANG = {
        "ja": {
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
        },
        "ja_minna": {
            "location": "[Most Important] Place names — towns, districts, neighborhoods, temples, shrines, bridges, streets in Edo (e.g., 神田, 浅草, 本所, 深川, 永代橋). The dominant entity comprising about 59% of the data",
            "damage": "[Important] Damage descriptions — collapsed buildings, fires, deaths, injuries, casualty counts (e.g., 潰家, 焼失, 死人, 怪我人, 大破, 半潰). About 27% of the data",
            "person": "[Normal] Person names — daimyo, officials, residents, victims (e.g., 井上筑後守, 黒田豊前守, 石川殿). About 12% of the data",
            "datetime": "[Somewhat Rare] Date/time expressions — era names, years, months, days, hours (e.g., 安政二年, 十月二日, 子の刻, 寅の上刻). About 2% of the data",
        },
        "de": {
            "PER": "Personennamen – vollständige Namen, Titel mit Namen, Adelstitel (z.B. 'König Friedrich Wilhelm', 'Herr Dr. Müller')",
            "LOC": "Ortsnamen – Städte, Länder, Regionen, Straßen, Gebäude (z.B. 'Berlin', 'Preußen', 'Friedrichstraße')",
            "ORG": "Organisationen – Behörden, Firmen, Vereine, Parteien (z.B. 'Königl. Akademie', 'Berliner Börse')",
            "HumanProd": "Menschliche Erzeugnisse – Bücher, Zeitungen, Kunstwerke, Gesetze (z.B. 'Allgemeine Zeitung', 'Handelsgesetzbuch')",
        },
        "fr": {
            "PER": "Noms de personnes – noms complets, titres avec noms, titres de noblesse (ex: 'M. le baron de Rothschild')",
            "LOC": "Noms de lieux – villes, pays, régions, rues, bâtiments (ex: 'Paris', 'Versailles')",
            "ORG": "Organisations – administrations, entreprises, associations (ex: 'Académie royale')",
            "HumanProd": "Productions humaines – livres, journaux, œuvres d'art, lois (ex: 'Le Moniteur universel')",
        },
        "en": {
            "PER": "Person names – full names, titles with names (e.g. 'Lord Wellington', 'Mr. Churchill')",
            "LOC": "Location names – cities, countries, regions, streets, buildings (e.g. 'London', 'Westminster')",
            "ORG": "Organizations – institutions, companies, associations (e.g. 'Royal Society', 'Parliament')",
            "HumanProd": "Human productions – books, newspapers, artworks, laws (e.g. 'The Times', 'Magna Carta')",
        },
    }

    SYSTEM_MESSAGES = {
        "ja": "You are an expert in Edo-period kabuki and Yakushahyoubanki literature.",
        "ja_minna": "You are an expert in late-Edo-period (Bakumatsu) Japanese historical documents, particularly disaster records, damage reports, and chronicles related to the 1855 Ansei Edo Earthquake and contemporaneous events. You can perfectly reproduce the cursive (kuzushiji) writing style, mixed kanji-hiragana orthography, and the formal register used in Edo-period administrative documents and damage chronicles.",
        "de": "Du bist ein Experte für historische deutsche Zeitungstexte des 17.–19. Jahrhunderts. Du kannst den Schreibstil historischer Zeitungen perfekt reproduzieren, einschließlich Frakturschrift, veralteter Orthographie und OCR-typischer Fehler.",
        "fr": "Vous êtes un expert en textes de presse historiques français du XVIIe au XIXe siècle. Vous pouvez reproduire parfaitement le style d'écriture des journaux historiques, y compris l'orthographe ancienne.",
        "en": "You are an expert in historical English newspaper texts from the 17th–19th century. You can perfectly reproduce the writing style of historical newspapers, including archaic spelling and OCR-typical errors.",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        provider: str = "openai",
        lang: str = "ja"
    ):
        """
        Initialize generation model

        Args:
            api_key: API key
            model: Model name
            temperature: Temperature parameter
            base_url: API base URL (OpenAI only)
            provider: API provider ("openai" or "claude")
            lang: Target language for prompt generation (ja/de/fr/en)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.provider = provider.lower()
        self.lang = lang
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
        Create generation prompt (dispatches by language)

        Args:
            corpus: Corpus
            target_entities: Target entities (entities that must be included)

        Returns:
            Prompt string
        """
        if self.lang == "ja":
            return self._create_prompt_ja(corpus, target_entities)
        elif self.lang == "ja_minna":
            return self._create_prompt_ja_minna(corpus, target_entities)
        else:
            return self._create_prompt_multilingual(corpus, target_entities)

    def _create_prompt_ja(
        self,
        corpus: NERSyntheticDataCorpus,
        target_entities: Optional[List[str]] = None
    ) -> str:
        """Create Japanese prompt (original Yakusha/Kabuki domain)"""
        entity_definitions = self.ENTITY_DEFINITIONS_BY_LANG.get("ja", {})

        entity_definitions_str = []
        for entity_type in sorted(corpus.entity_types):
            definition = entity_definitions.get(entity_type, "Related entity")
            examples = corpus.get_top_entities(entity_type, 5)
            examples_str = ", ".join(examples) if examples else "(no examples)"
            entity_definitions_str.append(
                f"- {entity_type} ({definition})\n  Examples: {examples_str}"
            )

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
                f"Example {i}:\nGenerated text: {sample.text}\nAnnotation:\n{chr(10).join(entities_list)}"
            )

        task_description = "Generate one passage related to Edo-period kabuki and Yakushahyoubanki, and identify and annotate the named entities (NER) in the passage."
        if target_entities:
            task_description += f"\n\nImportant: Please make sure to include the following entities in the passage: {', '.join(target_entities)}"

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

    def _create_prompt_ja_minna(
        self,
        corpus: NERSyntheticDataCorpus,
        target_entities: Optional[List[str]] = None
    ) -> str:
        """Create Japanese prompt for Minna domain (1855 Ansei Edo Earthquake records)."""
        entity_definitions = self.ENTITY_DEFINITIONS_BY_LANG.get("ja_minna", {})

        entity_definitions_str = []
        for entity_type in sorted(corpus.entity_types):
            definition = entity_definitions.get(entity_type, "Related entity")
            examples = corpus.get_top_entities(entity_type, 5)
            examples_str = ", ".join(examples) if examples else "(no examples)"
            entity_definitions_str.append(
                f"- {entity_type} ({definition})\n  Examples: {examples_str}"
            )

        # For minna, few-shot samples can have hundreds of entities which blows up the prompt.
        # Use SHORT text excerpts (<= 150 chars) and cap entities per example to keep prompt compact.
        few_shot_samples = corpus.get_few_shot_examples(n=3, min_entities=5)
        few_shot_examples = []
        MAX_TEXT_CHARS = 150
        MAX_ENTITIES_PER_EXAMPLE = 10

        for i, sample in enumerate(few_shot_samples, 1):
            # Take first MAX_TEXT_CHARS of text
            short_text = sample.text[:MAX_TEXT_CHARS]
            # Keep only entities that fit within short_text
            in_range_entities = [
                e for e in sample.entities
                if e['span'][1] <= len(short_text)
            ]
            # Cap at MAX_ENTITIES_PER_EXAMPLE
            selected_entities = in_range_entities[:MAX_ENTITIES_PER_EXAMPLE]

            entities_list = []
            for entity in selected_entities:
                entities_list.append(
                    f"    - [{entity['type']}] {entity['name']} "
                    f"(position: {entity['span'][0]}-{entity['span'][1]})"
                )
            if not entities_list:
                continue  # skip if no entities fit
            few_shot_examples.append(
                f"Example {i}:\nGenerated text: {short_text}\nAnnotation:\n{chr(10).join(entities_list)}"
            )

        task_description = (
            "Generate one passage in the style of late-Edo-period (Bakumatsu) Japanese damage records, "
            "earthquake chronicles, or administrative reports about the 1855 Ansei Edo Earthquake "
            "and contemporaneous events. Annotate all named entities (location, damage, person, datetime) in the passage."
        )
        if target_entities:
            task_description += f"\n\nImportant: Please make sure to include the following entities in the passage: {', '.join(target_entities)}"

        prompt = f"""# Domain Specification
You are an expert in late-Edo-period (Bakumatsu, ~1850s) Japanese historical documents,
particularly disaster records and damage chronicles related to the 1855 Ansei Edo Earthquake.
You can perfectly reproduce the cursive writing style, mixed kanji-hiragana orthography,
and the formal register used in Edo-period administrative documents and damage reports.

# Task
{task_description}

# Entity Types
The passage must contain the following types of entities:

{chr(10).join(entity_definitions_str)}

# Generation Guidelines
**Important: Generate in late-Edo-period (Bakumatsu) damage-record style.**

Writing style characteristics:
1. Mixed kanji and hiragana with historical kana orthography
2. Use of place markers like "辺", "町", "丁", "御", "様", "殿"
3. Damage vocabulary: 潰家 (collapsed houses), 焼失 (burned down), 大破 (heavy damage),
   半潰 (half-collapsed), 死人 (deceased), 怪我人 (injured), 焼亡 (destroyed by fire)
4. Honorific titles for officials: 〜守, 〜介, 〜殿, 〜様
5. Time expressions: era names (安政), traditional hours (子の刻, 寅の上刻), Japanese-style dates
6. Run-on style typical of damage reports — list-like enumeration of locations and damages
7. Length: approximately 80-200 characters

# Few-shot Examples
The following are actual examples from Edo-period earthquake records:

{chr(10).join(few_shot_examples)}

# Output Format
Please output in the following JSON format:

<start_annotation>
{{
    "text": "Generated late-Edo-period damage-record style passage",
    "entities": [
        {{
            "type": "entity type (location/damage/person/datetime)",
            "name": "entity text",
            "start": start position (integer),
            "end": end position (integer)
        }}
    ]
}}
</end_annotation>

Notes:
- In "text", write the generated late-Edo-period damage-record style passage
- start/end are character indices (starting from 0)
- Calculate entity positions accurately
- Use the four entity types exactly: location, damage, person, datetime

Now, please generate a new passage:"""
        return prompt

    def _create_prompt_multilingual(
        self,
        corpus: NERSyntheticDataCorpus,
        target_entities: Optional[List[str]] = None
    ) -> str:
        """Create prompt for multilingual historical newspaper NER (de/fr/en)"""
        lang = self.lang
        entity_definitions = self.ENTITY_DEFINITIONS_BY_LANG.get(lang, self.ENTITY_DEFINITIONS_BY_LANG["en"])

        # Build entity type definitions with examples from corpus
        entity_definitions_str = []
        for entity_type in sorted(corpus.entity_types):
            definition = entity_definitions.get(entity_type, entity_type)
            examples = corpus.get_top_entities(entity_type, 5)
            examples_str = ", ".join(examples) if examples else "(keine Beispiele)" if lang == "de" else "(no examples)"
            entity_definitions_str.append(
                f"- {entity_type}: {definition}\n  Examples: {examples_str}"
            )

        # Few-shot examples from real data
        few_shot_samples = corpus.get_few_shot_examples(n=3, min_entities=2)
        few_shot_examples = []
        for i, sample in enumerate(few_shot_samples, 1):
            entities_list = []
            for entity in sample.entities:
                entities_list.append(
                    f"    - [{entity['type']}] \"{entity['name']}\" "
                    f"(pos: {entity['span'][0]}-{entity['span'][1]})"
                )
            few_shot_examples.append(
                f"Example {i}:\nText: {sample.text}\nEntities:\n{chr(10).join(entities_list)}"
            )

        # Language-specific domain and style descriptions
        domain_desc = {
            "de": "Du bist ein Experte für historische deutsche Zeitungstexte des 17.–19. Jahrhunderts. Du kennst die typischen Merkmale von OCR-digitalisierten Zeitungstexten aus dieser Epoche.",
            "fr": "Vous êtes un expert en textes de presse historiques français du XVIIe au XIXe siècle. Vous connaissez les caractéristiques typiques des textes de journaux numérisés par OCR de cette époque.",
            "en": "You are an expert in historical English newspaper texts from the 17th–19th century. You know the typical characteristics of OCR-digitized newspaper texts from this era.",
        }

        style_guidelines = {
            "de": """Stilmerkmale:
1. Typische OCR-Fehler aus Frakturschrift: ſ statt s, f statt ſ, rn→m, ii→ü, cl→d
2. Veraltete Orthographie: th statt t (z.B. "thun"), ey statt ei, ä→ae
3. Historische Wortformen: "allhier", "desgleichen", "bishero", "anbey"
4. Lange Schachtelsätze mit Nebensätzen (typisch für 18./19. Jh.)
5. Höflichkeitsformeln: "Se. Königl. Majestät", "Ihro Durchlaucht"
6. Zeitungsstil: Nachrichtenberichte, Anzeigen, Bekanntmachungen
7. Themenbereiche: Politik, Handel, Kriege, Hofnachrichten, Theaterkritiken""",
            "fr": """Caractéristiques stylistiques:
1. Erreurs OCR typiques: ſ au lieu de s, lettres confondues
2. Orthographe ancienne: oi au lieu de ai (ex: "François" → "Françoiſ")
3. Formes historiques: "ledit", "icelle", "cy-devant"
4. Longues phrases avec subordonnées (typique du XVIIIe/XIXe siècle)
5. Formules de politesse: "Sa Majesté", "Son Excellence"
6. Style journalistique: nouvelles, annonces, avis officiels""",
            "en": """Style characteristics:
1. Typical OCR errors: ſ instead of s, rn→m, confused letters
2. Archaic spelling: "publick", "connexion", "honour"
3. Historical forms: "hath", "thereof", "aforesaid"
4. Long complex sentences typical of 18th/19th century prose
5. Honorifics: "His Majesty", "the Right Honourable"
6. Newspaper style: news reports, advertisements, official notices""",
        }

        task_description = f"Generate one realistic passage that resembles a historical newspaper text ({lang.upper()}, 17th–19th century) and annotate all named entities (NER) in the passage."
        if target_entities:
            task_description += f"\n\nIMPORTANT: You MUST include the following entities in the generated passage: {', '.join(target_entities)}"

        prompt = f"""# Domain
{domain_desc.get(lang, domain_desc['en'])}

# Task
{task_description}

# Entity Types
The passage must contain the following types of entities:

{chr(10).join(entity_definitions_str)}

# Writing Style Guidelines
{style_guidelines.get(lang, style_guidelines['en'])}

# Real Examples from the Corpus
{chr(10).join(few_shot_examples) if few_shot_examples else "(No examples available)"}

# Output Format
Output in the following JSON format:

<start_annotation>
{{
    "text": "Generated historical newspaper passage",
    "entities": [
        {{
            "type": "entity type",
            "name": "entity text as it appears in the passage",
            "start": start_position (integer, 0-indexed),
            "end": end_position (integer, exclusive)
        }}
    ]
}}
</end_annotation>

IMPORTANT:
- The text must read like a real 17th–19th century newspaper article
- start/end are character indices (0-indexed, end is exclusive)
- Calculate entity positions accurately – they must match the exact text
- Generate 50–150 words of text

Now generate a new passage:"""
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
        system_msg = self.SYSTEM_MESSAGES.get(self.lang, self.SYSTEM_MESSAGES["en"])
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def _generate_with_claude(self, prompt: str) -> Optional[str]:
        """Generate using Claude API"""
        system_msg = self.SYSTEM_MESSAGES.get(self.lang, self.SYSTEM_MESSAGES["en"])
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_msg,
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

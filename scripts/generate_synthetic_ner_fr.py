#!/usr/bin/env python3
"""Generate synthetic French NER data using ChatGPT API.

Mirrors the approach used for German NewsEye synthetic data:
- Extract top N entities per type from real data
- Ask ChatGPT to generate historical French newspaper articles
- Each article contains 2-5 entities with exact spans
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter
from openai import OpenAI

# ============================================================================
# Configuration
# ============================================================================
TOP_N = 20  # Top N entities per type to use as seeds
SAMPLES_PER_ENTITY = 5  # How many samples to generate per entity
BATCH_SIZE = 10  # Entities per API call
MODEL = "gpt-4o-mini"
OUTPUT_DIR = "/root/Ukiyo-e_NER/dataset/newseye_fr_synthetic_ner_dataset_chatgpt_N_20"
DATA_FILE = "/root/Ukiyo-e_NER/dataset/Newseye/fr/NewsEye-GT-NER_EL_StD-v1-test-fr.tsv"

SYSTEM_PROMPT = """You are an expert at generating synthetic training data for Named Entity Recognition (NER) in historical French newspapers from the 19th and early 20th century.

You will be given a list of named entities with their types. Generate realistic French newspaper articles (3-6 sentences each) that naturally incorporate these entities. The text should:
- Be written in historical French newspaper style (formal, period-appropriate language)
- Include the entities naturally in context
- Each article should contain 2-5 entities from the provided list

For EACH generated article, return a JSON object with:
- "text": the full article text
- "entities": array of objects, each with:
  - "name": exact entity text as it appears in the article
  - "span": [start_char_index, end_char_index] (0-indexed, exclusive end)
  - "type": one of "LOC", "PER", "ORG", "HumanProd"

CRITICAL: The span indices MUST be exact. Verify that text[start:end] == name.

Return a JSON array of 5 articles. Output ONLY valid JSON, no markdown or explanation."""


def extract_entities(data_file):
    """Extract entities from the French TSV file."""
    entities = {}
    cur_entity = []
    cur_type = None

    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_entity:
                    e = " ".join(cur_entity)
                    entities.setdefault(cur_type, []).append(e)
                    cur_entity = []
                    cur_type = None
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tok, tag = parts[0], parts[1]
            if tag.startswith("B-"):
                if cur_entity:
                    e = " ".join(cur_entity)
                    entities.setdefault(cur_type, []).append(e)
                cur_type = tag[2:]
                cur_entity = [tok]
            elif tag.startswith("I-") and cur_entity:
                cur_entity.append(tok)
            else:
                if cur_entity:
                    e = " ".join(cur_entity)
                    entities.setdefault(cur_type, []).append(e)
                    cur_entity = []
                    cur_type = None
        if cur_entity:
            e = " ".join(cur_entity)
            entities.setdefault(cur_type, []).append(e)

    # Get top N per type
    top_entities = {}
    for etype, elist in entities.items():
        c = Counter(elist)
        top_entities[etype] = [name for name, _ in c.most_common(TOP_N)]

    return top_entities


def validate_spans(article):
    """Validate and fix entity spans."""
    text = article["text"]
    valid_entities = []
    for ent in article.get("entities", []):
        name = ent["name"]
        start, end = ent["span"]

        # Check if span is correct
        if 0 <= start < len(text) and end <= len(text):
            if text[start:end] == name:
                valid_entities.append(ent)
                continue

        # Try to find the entity in text
        idx = text.find(name)
        if idx >= 0:
            ent["span"] = [idx, idx + len(name)]
            valid_entities.append(ent)
        else:
            print(f"  WARNING: Could not find '{name}' in text, skipping")

    article["entities"] = valid_entities
    return article


def generate_batch(client, entity_list, entity_types):
    """Generate a batch of synthetic articles."""
    # Build entity description
    entity_desc = []
    for name, etype in zip(entity_list, entity_types):
        entity_desc.append(f"- {name} ({etype})")
    entity_str = "\n".join(entity_desc)

    user_prompt = f"""Generate 5 historical French newspaper articles using these entities:

{entity_str}

Each article should use 2-5 of these entities. Return ONLY a JSON array."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            articles = json.loads(content)

            # Validate spans
            valid_articles = []
            for article in articles:
                article = validate_spans(article)
                if article["entities"]:  # Keep only if has valid entities
                    valid_articles.append(article)

            return valid_articles

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error (attempt {attempt+1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(5)

    return []


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("Extracting entities from French NewsEye data...")
    top_entities = extract_entities(DATA_FILE)

    for etype, elist in top_entities.items():
        print(f"  {etype}: {len(elist)} entities -> {elist[:5]}...")

    # Build entity pool with types
    all_entities = []
    all_types = []
    for etype, elist in top_entities.items():
        for name in elist:
            all_entities.append(name)
            all_types.append(etype)

    print(f"\nTotal seed entities: {len(all_entities)}")

    # Generate batches
    all_articles = []
    total_batches = (len(all_entities) * SAMPLES_PER_ENTITY) // BATCH_SIZE + 1

    print(f"Generating ~{total_batches} batches...")

    for batch_idx in range(total_batches):
        # Sample random subset of entities for this batch
        indices = random.sample(range(len(all_entities)), min(BATCH_SIZE, len(all_entities)))
        batch_entities = [all_entities[i] for i in indices]
        batch_types = [all_types[i] for i in indices]

        print(f"\n[Batch {batch_idx+1}/{total_batches}] Entities: {batch_entities[:3]}...")
        articles = generate_batch(client, batch_entities, batch_types)
        all_articles.extend(articles)
        print(f"  Generated {len(articles)} articles (total: {len(all_articles)})")

        time.sleep(1)  # Rate limiting

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "generated_ner_balanced.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    # Stats
    type_count = Counter()
    for article in all_articles:
        for ent in article["entities"]:
            type_count[ent["type"]] += 1

    print(f"\n{'='*50}")
    print(f"Generated {len(all_articles)} articles")
    print(f"Entity distribution: {dict(type_count)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic AJMC DE NER data using ChatGPT API.
AJMC = Ajax Multi-Commentary (classical studies, German text about Greek tragedies)
"""

import json
import os
import random
import re
import time
from collections import Counter
from openai import OpenAI

TOP_N = 20
MODEL = "gpt-4o-mini"
OUTPUT_DIR = "/root/Ukiyo-e_NER/dataset/ajmc_de_synthetic_ner_dataset_chatgpt_N_20"
DATA_FILE = "/root/Ukiyo-e_NER/dataset/AJMC/de/HIPE-2022-v2.1-ajmc-test-de.tsv"

SYSTEM_PROMPT = """You are an expert at generating synthetic training data for Named Entity Recognition (NER) in German classical philology commentaries about ancient Greek literature (especially Sophocles' Ajax).

You will be given a list of named entities with their types. Generate realistic German commentary text (3-6 sentences) in the style of 19th century classical philology. The text should:
- Discuss ancient Greek literary works, characters, and passages
- Use old German orthography (ſ for s where appropriate)
- Include passage/line references (e.g. "V. 123", "1135 f.", "vgl. 406")
- Reference classical authors and their works using standard abbreviations

Entity types:
- pers: Person names (mythological figures, ancient authors, scholars) e.g. Aias, Sophokles, Homer
- work: Literary work titles/abbreviations e.g. Ant., Phil., El., Od.
- scope: Line/passage references e.g. 1135, 406, 983 f.

For EACH generated text, return a JSON object with:
- "text": the full text
- "entities": array of objects, each with:
  - "name": exact entity text as it appears
  - "span": [start_char_index, end_char_index] (0-indexed, exclusive end)
  - "type": one of "pers", "work", "scope"

CRITICAL: The span indices MUST be exact. Verify that text[start:end] == name.

Return a JSON array of 5 texts. Output ONLY valid JSON, no markdown."""


def extract_entities(data_file):
    entities = {}
    cur_ent = None
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_ent:
                    entities.setdefault(cur_ent[1], []).append(cur_ent[0])
                    cur_ent = None
                continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            tok, tag = parts[0], parts[1]
            if tag.startswith('B-'):
                if cur_ent:
                    entities.setdefault(cur_ent[1], []).append(cur_ent[0])
                cur_ent = [tok, tag[2:]]
            elif tag.startswith('I-') and cur_ent:
                cur_ent[0] += ' ' + tok
            else:
                if cur_ent:
                    entities.setdefault(cur_ent[1], []).append(cur_ent[0])
                    cur_ent = None
    if cur_ent:
        entities.setdefault(cur_ent[1], []).append(cur_ent[0])

    top = {}
    for etype, elist in entities.items():
        if etype in ('loc', 'object'):  # too few
            continue
        c = Counter(elist)
        top[etype] = [name for name, _ in c.most_common(TOP_N)]
    return top


def validate_spans(article):
    text = article["text"]
    valid = []
    for ent in article.get("entities", []):
        name = ent["name"]
        start, end = ent["span"]
        if 0 <= start < len(text) and end <= len(text) and text[start:end] == name:
            valid.append(ent)
            continue
        idx = text.find(name)
        if idx >= 0:
            ent["span"] = [idx, idx + len(name)]
            valid.append(ent)
    article["entities"] = valid
    return article


def generate_batch(client, entity_list, entity_types):
    entity_desc = "\n".join(f"- {name} ({etype})" for name, etype in zip(entity_list, entity_types))
    user_prompt = f"Generate 5 German classical philology commentary texts using these entities:\n\n{entity_desc}\n\nEach text should use 2-5 of these entities. Return ONLY a JSON array."

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8, max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            articles = json.loads(content)
            valid = []
            for a in articles:
                a = validate_spans(a)
                if a["entities"]:
                    valid.append(a)
            return valid
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return []


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("Extracting entities from AJMC DE...")
    top = extract_entities(DATA_FILE)
    for t, elist in top.items():
        print(f"  {t}: {len(elist)} -> {elist[:5]}...")

    all_entities, all_types = [], []
    for etype, elist in top.items():
        for name in elist:
            all_entities.append(name)
            all_types.append(etype)

    print(f"Total seed entities: {len(all_entities)}")
    all_articles = []
    total_batches = max(40, (len(all_entities) * 5) // 10 + 1)
    print(f"Generating {total_batches} batches...")

    for i in range(total_batches):
        indices = random.sample(range(len(all_entities)), min(10, len(all_entities)))
        batch_e = [all_entities[j] for j in indices]
        batch_t = [all_types[j] for j in indices]
        print(f"[{i+1}/{total_batches}] {batch_e[:3]}...")
        articles = generate_batch(client, batch_e, batch_t)
        all_articles.extend(articles)
        print(f"  +{len(articles)} (total: {len(all_articles)})")
        time.sleep(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "generated_ner_balanced.json"), "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    tc = Counter()
    for a in all_articles:
        for e in a["entities"]:
            tc[e["type"]] += 1
    print(f"\nGenerated {len(all_articles)} articles. Entities: {dict(tc)}")


if __name__ == "__main__":
    main()

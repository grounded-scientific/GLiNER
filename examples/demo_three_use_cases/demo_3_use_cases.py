#!/usr/bin/env python3
"""
Inside the Code (Episode 1): GLiNER 3 Practical CPU Use Cases Demo
Author: Alex MacIntosh, PhD | Grounded Scientific

Demonstrates:
1. Custom Zero-Shot NER (Domain-Specific Labels + Exact Character Anchoring)
2. Local Privacy-Safe PII Detection & Redaction
3. Single-Pass Knowledge Graph Construction (GLiNER-Relex Joint NER + RE)
"""

import argparse
import time

from gliner import GLiNER


def load(checkpoint: str) -> GLiNER:
    """Load a checkpoint on CPU and report load time (kept separate from inference timing)."""
    print(f"[*] Loading {checkpoint} (CPU)...")
    t0 = time.perf_counter()
    model = GLiNER.from_pretrained(checkpoint).to("cpu")
    print(f"[+] Loaded in {time.perf_counter() - t0:.2f}s\n")
    return model


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def demo_1_custom_zero_shot_ner() -> None:
    print_banner("USE CASE 1: Custom Zero-Shot NER & Verbatim Quote Anchoring")
    checkpoint = "urchade/gliner_small-v2.1"
    model = load(checkpoint)

    text = (
        "Acme Analytics built a retrieval pipeline for Northwind Pharmaceuticals "
        "in Toronto, Ontario, validating GLP-1 receptor agonist clinical trial endpoints "
        "under ISO 27001 security standards."
    )
    labels = [
        "ai_consulting_firm",
        "client_organization",
        "location",
        "biomedical_target",
        "governance_standard",
    ]

    print(f"Input Text:\n  \"{text}\"\n")
    print(f"Custom Zero-Shot Labels: {labels}\n")

    t1 = time.perf_counter()
    entities = model.predict_entities(text, labels, threshold=0.35)
    elapsed_ms = (time.perf_counter() - t1) * 1000

    print(f"Extracted {len(entities)} Entities in {elapsed_ms:.1f} ms:")
    for ent in entities:
        start, end = ent["start"], ent["end"]
        anchor_ok = text[start:end] == ent["text"]
        print(
            f"  - [{ent['label']:<20}] {ent['text']:<26} "
            f"(score: {ent['score']:.2f} | span: [{start}:{end}] | anchor_verified={anchor_ok})"
        )


def demo_2_local_pii_redaction() -> None:
    print_banner("USE CASE 2: Local, Privacy-Safe PII Detection & Span Redaction")
    checkpoint = "urchade/gliner_multi_pii-v1"
    model = load(checkpoint)

    raw_ticket = (
        "Patient Jane Doe (DOB: 1984-07-19, Health Card: 4928-331-882-XK) "
        "called from +1 519-555-0199 and emailed jane.doe@clinic.ca regarding "
        "invoice #INV-99210."
    )
    pii_labels = [
        "person",
        "date_of_birth",
        "health_card_number",
        "phone_number",
        "email",
    ]

    t0 = time.perf_counter()
    spans = model.predict_entities(raw_ticket, pii_labels, threshold=0.4)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Redact in reverse character-offset order so earlier indices never shift
    redacted = raw_ticket
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        tag = f"[REDACTED:{span['label'].upper()}]"
        redacted = redacted[: span["start"]] + tag + redacted[span["end"] :]

    print(f"Original Text:\n  {raw_ticket}\n")
    print(f"Detected {len(spans)} PII Spans in {elapsed_ms:.1f} ms:")
    for s in spans:
        print(f"  - {s['text']:<22} => {s['label']} (score: {s['score']:.2f})")
    print(f"\nRedacted Output:\n  {redacted}")


def demo_3_knowledge_graph_relex() -> None:
    print_banner("USE CASE 3: Single-Pass Knowledge Graph Construction (GLiNER-Relex)")
    checkpoint = "knowledgator/gliner-relex-large-v1.0"
    model = load(checkpoint)

    text = (
        "Urchade Zaratiana developed GLiNER at Université Sorbonne Paris Nord in 2024. "
        "Knowledgator Engineering later introduced GLiNER-Relex for building GraphRAG pipelines."
    )
    entity_labels = ["researcher", "ai_framework", "organization", "university", "date"]
    relation_labels = ["developed", "affiliated_with", "introduced", "used_for"]

    t0 = time.perf_counter()
    entities_batch, relations_batch = model.inference(
        texts=[text],
        labels=entity_labels,
        relations=relation_labels,
        threshold=0.35,
        relation_threshold=0.40,
        return_relations=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    entities = entities_batch[0]
    relations = relations_batch[0]

    print(f"Input Text:\n  \"{text}\"\n")
    print(f"Single Forward Pass completed in {elapsed_ms:.1f} ms")
    print(f"\nNodes ({len(entities)} Entities):")
    for ent in entities:
        print(f"  (Node) {ent['text']:<32} : {ent['label']} [{ent['start']}:{ent['end']}]")

    print(f"\nDirected Edges ({len(relations)} Knowledge Graph Triplets):")
    for rel in relations:
        head = rel["head"]
        tail = rel["tail"]
        env_start = min(head["start"], tail["start"])
        env_end = max(head["end"], tail["end"])
        snippet = text[env_start:env_end]
        print(
            f"  (Edge) {head['text']} --[{rel['relation']} ({rel['score']:.2f})]--> {tail['text']}\n"
            f"         Provenance Anchor [{env_start}:{env_end}]: \"{snippet}\""
        )


DEMOS = {
    "1": demo_1_custom_zero_shot_ner,
    "2": demo_2_local_pii_redaction,
    "3": demo_3_knowledge_graph_relex,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=DEMOS, nargs="+", help="run only these use cases (default: all)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between use cases (live presenting)")
    args = ap.parse_args()
    for key in args.only or DEMOS:
        DEMOS[key]()
        if args.pause:
            input("\n[Enter] to continue...")

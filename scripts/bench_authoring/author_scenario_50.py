#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Author 50 scenario-based evaluation questions for Vesta Benchmark v2.

Categories:
1. concept_lookup (scene-0001 .. scene-0010)
2. comparative (scene-0011 .. scene-0020)
3. procedural (scene-0021 .. scene-0030)
4. complex_explanation (scene-0031 .. scene-0040)
5. adversarial_abstention (scene-0041 .. scene-0050)
"""

from __future__ import annotations

import json
from pathlib import Path

from libzim.reader import Archive

from vesta.zim.extract import extract_article
from vesta.zim.reader import read_entry_sync

ZIM = "data/zims/wikipedia_en_top_nopic_2026-06.zim"


def get_text(archive: Archive, path: str) -> str:
    raw = read_entry_sync(archive, path)
    if raw.is_redirect:
        raw = read_entry_sync(archive, raw.redirect_target)
    return extract_article(raw.content, path=raw.path, title=raw.title).text


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def build_questions() -> list[dict]:  # noqa: PLR0915
    archive = Archive(ZIM)

    def check(path: str, fact: str) -> str:
        text = get_text(archive, path)
        norm_fact = normalize(fact)
        norm_text = normalize(text)
        if norm_fact not in norm_text:
            raise ValueError(f"Fact {fact!r} not found in article {path!r}")
        return fact

    questions = []

    # =========================================================================
    # 1. CONCEPT LOOKUP (scene-0001 .. scene-0010)
    # =========================================================================

    # scene-0001: Dunning-Kruger effect
    questions.append(
        {
            "id": "scene-0001",
            "question": "What is the name of the cognitive bias where people with low ability in a particular domain systematically overestimate their competence?",
            "capability": "concept_lookup",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "psychology", "cognitive_bias"],
            "expected_behavior": "answer",
            "answer": "The Dunning–Kruger effect.",
            "answer_detail": "The Dunning–Kruger effect is a cognitive bias in which people with low ability in a specific area give overly positive assessments of their ability.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Dunning–Kruger effect",
                    "article_path": "Dunning–Kruger_effect",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Dunning–Kruger_effect", "cognitive bias"), "source_index": 0},
                {
                    "fact": check(
                        "Dunning–Kruger_effect",
                        "low ability in a specific area to give overly positive assessments of this ability",
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0002: Bystander effect
    questions.append(
        {
            "id": "scene-0002",
            "question": "What is the social psychological phenomenon where individuals are less likely to offer assistance to a victim when other people are present?",
            "capability": "concept_lookup",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "psychology", "social_psychology"],
            "expected_behavior": "answer",
            "answer": "The bystander effect (or bystander apathy).",
            "answer_detail": "The bystander effect is a social psychological theory stating that individuals are less likely to offer help to a victim in the presence of other people, often driven by diffusion of responsibility.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Bystander effect",
                    "article_path": "Bystander_effect",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Bystander_effect", "social psychological theory"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Bystander_effect",
                        "less likely to offer help to a victim in the presence of other people",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Bystander_effect", "diffusion of responsibility"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0003: Pareidolia
    questions.append(
        {
            "id": "scene-0003",
            "question": "What is the psychological tendency to perceive meaningful images or recognizable shapes—such as human faces—in ambiguous or random visual stimuli like clouds or rock formations?",
            "capability": "concept_lookup",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "psychology", "perception"],
            "expected_behavior": "answer",
            "answer": "Pareidolia.",
            "answer_detail": "Pareidolia is the tendency for perception to impose a meaningful interpretation on a nebulous stimulus, such as interpreting random images or patterns of light and shadow as faces or objects in cloud formations.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Pareidolia",
                    "article_path": "Pareidolia",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Pareidolia",
                        "tendency for perception to impose a meaningful interpretation on a nebulous stimulus",
                    ),
                    "source_index": 0,
                },
                {"fact": check("Pareidolia", "faces in inanimate objects"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0004: Streisand effect
    questions.append(
        {
            "id": "scene-0004",
            "question": "What phenomenon occurs when an attempt to hide, remove, or suppress information backfires and inadvertently leads to widespread public awareness of that information?",
            "capability": "concept_lookup",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "sociology", "internet_culture"],
            "expected_behavior": "answer",
            "answer": "The Streisand effect.",
            "answer_detail": "The Streisand effect is the phenomenon in which an attempt to hide, remove, or censor information results in the unintended consequence of increasing public awareness of the information.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Streisand effect",
                    "article_path": "Streisand_effect",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Streisand_effect", "attempt to hide, remove, or censor information"
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Streisand_effect",
                        "unintended consequence of the effort instead increasing public awareness",
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0005: Confirmation bias
    questions.append(
        {
            "id": "scene-0005",
            "question": "What cognitive bias involves searching for, interpreting, favoring, and recalling information in a way that aligns with and reinforces one's preexisting beliefs or hypotheses?",
            "capability": "concept_lookup",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "psychology", "cognitive_bias"],
            "expected_behavior": "answer",
            "answer": "Confirmation bias.",
            "answer_detail": "Confirmation bias is the tendency to search for, interpret, favor and recall information in a way that confirms or supports one's prior beliefs, values, or decisions.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Confirmation bias",
                    "article_path": "Confirmation_bias",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Confirmation_bias",
                        "tendency to search for, interpret, favor and recall information",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Confirmation_bias", "confirms or supports one's prior beliefs"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0006: Ship of Theseus
    questions.append(
        {
            "id": "scene-0006",
            "question": "What famous philosophical thought experiment questions whether an entity that has had every single one of its original parts replaced over time remains fundamentally the identical object?",
            "capability": "concept_lookup",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "philosophy", "metaphysics"],
            "expected_behavior": "answer",
            "answer": "The Ship of Theseus (or Theseus's paradox).",
            "answer_detail": "The Ship of Theseus is a thought experiment in metaphysics regarding whether an object is the same object after having all of its original components replaced with others over time, famously preserved by Plutarch.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Ship of Theseus",
                    "article_path": "Ship_of_Theseus",
                    "fact_location": "lede and background",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Ship_of_Theseus", "thought experiment about whether an object"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Ship_of_Theseus",
                        "all of its original components replaced with others over time",
                    ),
                    "source_index": 0,
                },
                {"fact": check("Ship_of_Theseus", "Plutarch"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0007: Survivorship bias
    questions.append(
        {
            "id": "scene-0007",
            "question": "What statistical error occurs when an analysis focuses only on entities that successfully passed a filter or selection process while ignoring those that failed to pass, leading to distorted conclusions?",
            "capability": "concept_lookup",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "statistics", "cognitive_bias"],
            "expected_behavior": "answer",
            "answer": "Survivorship bias (or survival bias).",
            "answer_detail": "Survivorship bias is a statistical error that results from concentrating on entities that passed a selection process while overlooking those that did not, famously analyzed by statistician Abraham Wald during World War II regarding aircraft armor.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Survivorship bias",
                    "article_path": "Survivorship_bias",
                    "fact_location": "lede and Military section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Survivorship_bias",
                        "statistical error that results from concentrating on entities that passed a selection process",
                    ),
                    "source_index": 0,
                },
                {"fact": check("Survivorship_bias", "Abraham Wald"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0008: Cognitive dissonance
    questions.append(
        {
            "id": "scene-0008",
            "question": "What psychological concept, formulated by Leon Festinger, describes the mental discomfort experienced when holding contradictory beliefs, ideas, or values simultaneously?",
            "capability": "concept_lookup",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "psychology"],
            "expected_behavior": "answer",
            "answer": "Cognitive dissonance.",
            "answer_detail": "Cognitive dissonance is the mental discomfort experienced when an action or idea is psychologically inconsistent with another belief or value, formulated by Leon Festinger.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Cognitive dissonance",
                    "article_path": "Cognitive_dissonance",
                    "fact_location": "lede and Overview",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Cognitive_dissonance", "Leon Festinger"), "source_index": 0},
                {
                    "fact": check("Cognitive_dissonance", "psychologically inconsistent"),
                    "source_index": 0,
                },
                {"fact": check("Cognitive_dissonance", "mental discomfort"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0009: Tragedy of the commons
    questions.append(
        {
            "id": "scene-0009",
            "question": "What economic and environmental concept describes how individual actors with open access to a shared resource deplete or spoil that resource through uncoordinated self-interest?",
            "capability": "concept_lookup",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "economics", "ecology"],
            "expected_behavior": "answer",
            "answer": "The tragedy of the commons.",
            "answer_detail": "The tragedy of the commons, explored in a 1968 essay by Garrett Hardin, is the concept that in a system where individuals benefit from the use of a shared resource while the cost of that use is shared by all, rational self-interest leads to the depletion of the resource.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Tragedy of the commons",
                    "article_path": "Tragedy_of_the_commons",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Tragedy_of_the_commons", "Garrett Hardin"), "source_index": 0},
                {"fact": check("Tragedy_of_the_commons", "shared resource"), "source_index": 0},
                {
                    "fact": check("Tragedy_of_the_commons", "depletion of the resource"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0010: Prisoner's dilemma
    questions.append(
        {
            "id": "scene-0010",
            "question": "What foundational game theory thought experiment shows that two rational individuals might both choose to defect rather than cooperate, even though mutual cooperation yields a superior collective outcome?",
            "capability": "concept_lookup",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "concept_lookup", "game_theory", "economics"],
            "expected_behavior": "answer",
            "answer": "The prisoner's dilemma.",
            "answer_detail": "In game theory, the prisoner's dilemma is a thought experiment involving two rational agents who can either cooperate for mutual benefit or betray their partner (defect), originally designed by Merrill Flood and Melvin Dresher in 1950.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Prisoner's dilemma",
                    "article_path": "Prisoner's_dilemma",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Prisoner's_dilemma", "thought experiment involving two rational agents"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Prisoner's_dilemma", "Merrill Flood"), "source_index": 0},
                {
                    "fact": check(
                        "Prisoner's_dilemma", "cooperation yields a higher payoff for each"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "concept_lookup",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # =========================================================================
    # 2. COMPARATIVE (scene-0011 .. scene-0020)
    # =========================================================================

    # scene-0011: Mitosis vs Meiosis
    questions.append(
        {
            "id": "scene-0011",
            "question": "How do mitosis and meiosis differ in terms of the number of daughter cells produced and whether the resulting cells are genetically identical to the parent cell?",
            "capability": "comparative",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "comparative", "biology", "genetics"],
            "expected_behavior": "answer",
            "answer": "Mitosis produces two genetically identical daughter cells, whereas meiosis produces four genetically distinct haploid daughter cells.",
            "answer_detail": "Mitosis divides a mother cell into two daughter cells that are genetically identical to each other and the parent, while meiosis involves two rounds of division to produce four daughter cells, each with half the number of chromosomes (haploid) and undergoing genetic recombination.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Mitosis",
                    "article_path": "Mitosis",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Meiosis",
                    "article_path": "Meiosis",
                    "fact_location": "lede",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Mitosis", "two daughter cells genetically identical to each other"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Meiosis", "four daughter cells"), "source_index": 1},
                {"fact": check("Meiosis", "haploid"), "source_index": 1},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0012: Nuclear fission vs Nuclear fusion
    questions.append(
        {
            "id": "scene-0012",
            "question": "What is the fundamental physical difference between nuclear fission and nuclear fusion regarding how atomic nuclei are altered to release energy?",
            "capability": "comparative",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "comparative", "physics", "nuclear_physics"],
            "expected_behavior": "answer",
            "answer": "Nuclear fission splits a heavy atomic nucleus into smaller fragments, whereas nuclear fusion combines two or more lighter atomic nuclei into a larger nucleus.",
            "answer_detail": "In nuclear fission, fissile heavy nuclides split during interactions to sustain a nuclear chain reaction, whereas in nuclear fusion, light atomic nuclei combine to form a larger nucleus at high temperatures.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Nuclear fission",
                    "article_path": "Nuclear_fission",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Nuclear fusion",
                    "article_path": "Nuclear_fusion",
                    "fact_location": "lede",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check("Nuclear_fission", "fissile nuclides easily split"),
                    "source_index": 0,
                },
                {"fact": check("Nuclear_fission", "nuclear chain reaction"), "source_index": 0},
                {
                    "fact": check("Nuclear_fusion", "combine to form a larger nucleus"),
                    "source_index": 1,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0013: Alternating current vs Direct current
    questions.append(
        {
            "id": "scene-0013",
            "question": "How do alternating current (AC) and direct current (DC) differ in the directional behavior of their electric charge flow?",
            "capability": "comparative",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "comparative", "physics", "electrical_engineering"],
            "expected_behavior": "answer",
            "answer": "Alternating current periodically reverses direction, whereas direct current flows in a single constant direction.",
            "answer_detail": "Alternating current (AC) periodically reverses direction and changes its magnitude continuously, and its voltage can be easily converted with transformers. Direct current (DC), produced by sources like batteries or converted from AC via rectifiers, flows continuously in one direction.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Alternating current",
                    "article_path": "Alternating_current",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Direct current",
                    "article_path": "Direct_current",
                    "fact_location": "lede and conversion",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check("Alternating_current", "periodically reverses direction"),
                    "source_index": 0,
                },
                {"fact": check("Alternating_current", "transformer"), "source_index": 0},
                {"fact": check("Direct_current", "rectifier"), "source_index": 1},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0014: DNA vs RNA
    questions.append(
        {
            "id": "scene-0014",
            "question": "What are the primary chemical differences between DNA and RNA regarding their sugar component and their standard pyrimidine nitrogenous bases?",
            "capability": "comparative",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "comparative", "biochemistry", "molecular_biology"],
            "expected_behavior": "answer",
            "answer": "DNA contains deoxyribose sugar and thymine, whereas RNA contains ribose sugar and uracil.",
            "answer_detail": "DNA nucleotides contain deoxyribose and the bases adenine, cytosine, guanine, and thymine, forming a double helix. RNA nucleotides contain ribose sugar and use uracil instead of thymine, typically forming a single-stranded molecule.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "DNA",
                    "article_path": "DNA",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "RNA",
                    "article_path": "RNA",
                    "fact_location": "lede and structure",
                    "required": True,
                },
            ],
            "sub_facts": [
                {"fact": check("DNA", "sugar called deoxyribose"), "source_index": 0},
                {"fact": check("DNA", "thymine"), "source_index": 0},
                {"fact": check("RNA", "ribose sugar"), "source_index": 1},
                {"fact": check("RNA", "uracil"), "source_index": 1},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0015: Prokaryote vs Eukaryote
    questions.append(
        {
            "id": "scene-0015",
            "question": "What is the defining structural difference in cellular organization between prokaryotic and eukaryotic organisms?",
            "capability": "comparative",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "comparative", "biology", "cell_biology"],
            "expected_behavior": "answer",
            "answer": "Prokaryotes lack a membrane-bound nucleus and membrane-bound organelles, whereas eukaryotes have a membrane-bound nucleus and membrane-bound organelles.",
            "answer_detail": "A prokaryote's cell lacks a nucleus or other membrane-bound organelles (comprising bacteria and archaea), while eukaryotic cells (plants, animals, fungi) contain a membrane-bound nucleus and specialized organelles.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Prokaryote",
                    "article_path": "Prokaryote",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Eukaryote",
                    "article_path": "Eukaryote",
                    "fact_location": "lede",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Prokaryote", "lacks a nucleus or other membrane-bound organelles"
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Eukaryote", "cells have a membrane-bound nucleus"),
                    "source_index": 1,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0016: Steam engine vs Internal combustion engine
    questions.append(
        {
            "id": "scene-0016",
            "question": "How do steam engines and internal combustion engines differ in where fuel combustion occurs relative to the working fluid and cylinders?",
            "capability": "comparative",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "comparative", "engineering", "thermodynamics"],
            "expected_behavior": "answer",
            "answer": "Steam engines are external combustion engines where fuel is burned outside the working fluid cylinder, whereas internal combustion engines burn fuel inside an integral combustion chamber.",
            "answer_detail": "A defining feature of steam engines is that they are external combustion engines where the working fluid (steam) is separated from the combustion products. In contrast, an internal combustion engine burns fuel directly with an oxidizer inside a combustion chamber that forms an integral part of the working fluid circuit.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Steam engine",
                    "article_path": "Steam_engine",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Internal combustion engine",
                    "article_path": "Internal_combustion_engine",
                    "fact_location": "lede",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Steam_engine",
                        "external combustion engines, where the working fluid is separated from the combustion products",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Internal_combustion_engine",
                        "combustion chamber that is an integral part of the working fluid flow circuit",
                    ),
                    "source_index": 1,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0017: Hard disk drive vs Solid-state drive
    questions.append(
        {
            "id": "scene-0017",
            "question": "How do hard disk drives (HDDs) and solid-state drives (SSDs) fundamentally differ in their physical storage media and mechanical operation?",
            "capability": "comparative",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "comparative", "computer_hardware", "storage"],
            "expected_behavior": "answer",
            "answer": "HDDs store data magnetically on rapidly rotating mechanical platters using moving read/write heads, while SSDs store data electronically in solid-state flash memory with no moving parts.",
            "answer_detail": "HDDs use magnetic storage on rigid rapidly rotating platters accessed by moving read/write heads on a spindle. SSDs utilize semiconductor NAND flash memory assemblies to store data persistently without any moving mechanical parts.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Hard disk drive",
                    "article_path": "Hard_disk_drive",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Solid-state drive",
                    "article_path": "Solid-state_drive",
                    "fact_location": "lede",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Hard_disk_drive", "rotating platters coated with magnetic material"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Solid-state_drive", "flash memory"), "source_index": 1},
                {
                    "fact": check("Solid-state_drive", "SSDs have no moving parts"),
                    "source_index": 1,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0018: Artery vs Vein
    questions.append(
        {
            "id": "scene-0018",
            "question": "How do arteries and veins differ regarding the direction of blood flow relative to the heart and the presence of internal one-way valves?",
            "capability": "comparative",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "comparative", "anatomy", "cardiovascular"],
            "expected_behavior": "answer",
            "answer": "Arteries carry blood away from the heart under higher blood pressure, whereas veins return blood towards the heart and possess one-way venous valves to prevent backflow.",
            "answer_detail": "Arteries transport oxygenated blood away from the heart in the systemic circulation under high pressure, while veins carry blood towards the heart at lower pressure and are equipped with one-way valves to maintain unidirectional flow.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Artery",
                    "article_path": "Artery",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Vein",
                    "article_path": "Vein",
                    "fact_location": "lede and valves",
                    "required": True,
                },
            ],
            "sub_facts": [
                {"fact": check("Artery", "blood away from the heart"), "source_index": 0},
                {
                    "fact": check(
                        "Artery", "blood pressure higher than other parts of the circulatory system"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Vein", "blood towards the heart"), "source_index": 1},
                {
                    "fact": check(
                        "Vein", "one-way (unidirectional) venous valves to prevent backflow"
                    ),
                    "source_index": 1,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0019: Innate vs Adaptive immune system
    questions.append(
        {
            "id": "scene-0019",
            "question": "How do the innate and adaptive immune systems differ in antigen specificity and their ability to generate long-term immunological memory?",
            "capability": "comparative",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "comparative", "immunology", "medicine"],
            "expected_behavior": "answer",
            "answer": "The innate immune system provides non-specific defenses without generating immunological memory, whereas the adaptive immune system mounts an antigen-specific response and creates long-term immunological memory.",
            "answer_detail": "Innate immunity uses generic mechanisms like phagocytes (neutrophils) and non-specific defenses, whereas adaptive immunity relies on T cells and B cells to produce antigen-specific antibodies and establish lasting immunological memory against pathogens.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Innate immune system",
                    "article_path": "Innate_immune_system",
                    "fact_location": "lede and defenses",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Adaptive immune system",
                    "article_path": "Adaptive_immune_system",
                    "fact_location": "lede",
                    "required": True,
                },
            ],
            "sub_facts": [
                {"fact": check("Innate_immune_system", "non-specific defenses"), "source_index": 0},
                {
                    "fact": check(
                        "Adaptive_immune_system", "Adaptive immunity creates immunological memory"
                    ),
                    "source_index": 1,
                },
                {
                    "fact": check(
                        "Adaptive_immune_system",
                        "Antibodies are a critical part of the adaptive immune system",
                    ),
                    "source_index": 1,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0020: Cold fusion vs Nuclear fusion
    questions.append(
        {
            "id": "scene-0020",
            "question": "How did the hypothesized cold fusion phenomenon proposed by Pons and Fleischmann differ from conventional thermonuclear fusion regarding operating temperature conditions?",
            "capability": "comparative",
            "difficulty": "hard",
            "slice": "core",
            "tags": ["scenario", "comparative", "physics", "nuclear_physics"],
            "expected_behavior": "answer",
            "answer": "Cold fusion hypothesized nuclear fusion occurring at or near room temperature, whereas conventional thermonuclear fusion requires extreme temperatures of millions of degrees to overcome the Coulomb barrier.",
            "answer_detail": "Cold fusion was hypothesized as a nuclear reaction occurring at or near room temperature on palladium electrodes, contrasting with standard nuclear fusion where high temperature plasma is required for nuclei to overcome the Coulomb barrier.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Cold fusion",
                    "article_path": "Cold_fusion",
                    "fact_location": "lede",
                    "required": True,
                },
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Nuclear fusion",
                    "article_path": "Nuclear_fusion",
                    "fact_location": "lede and Coulomb barrier",
                    "required": True,
                },
            ],
            "sub_facts": [
                {
                    "fact": check("Cold_fusion", "occur at, or near, room temperature"),
                    "source_index": 0,
                },
                {"fact": check("Cold_fusion", "Pons and Fleischmann"), "source_index": 0},
                {
                    "fact": check("Nuclear_fusion", 'fusing" using high temperatures'),
                    "source_index": 1,
                },
                {"fact": check("Nuclear_fusion", "Coulomb barrier"), "source_index": 1},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "comparative",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # =========================================================================
    # 3. PROCEDURAL (scene-0021 .. scene-0030)
    # =========================================================================

    # scene-0021: CPR
    questions.append(
        {
            "id": "scene-0021",
            "question": "What is the recommended chest compression rate per minute and target compression depth for adult cardiopulmonary resuscitation (CPR)?",
            "capability": "procedural",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "procedural", "first_aid", "emergency_medicine"],
            "expected_behavior": "answer",
            "answer": "A compression rate of 100 to 120 compressions per minute and a depth of at least 5 cm (2 inches).",
            "answer_detail": "High-quality adult CPR guidelines specify delivering chest compressions at a rate of 100 to 120 compressions per minute with adequate compression depth to maintain vital perfusion.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Cardiopulmonary resuscitation",
                    "article_path": "Cardiopulmonary_resuscitation",
                    "fact_location": "Compression technique guidelines",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Cardiopulmonary_resuscitation", "100 to 120 per minute"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Cardiopulmonary_resuscitation", "100–120 compressions per minute"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0022: Burn first aid
    questions.append(
        {
            "id": "scene-0022",
            "question": "What is the recommended immediate first-aid cooling procedure for a thermal burn, and why should ice water be avoided?",
            "capability": "procedural",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "procedural", "first_aid", "emergency_medicine"],
            "expected_behavior": "answer",
            "answer": "Cool the burn using cool tap water (10–25 °C / 50–77 °F); do not use ice water, as it can worsen tissue damage or cause hypothermia.",
            "answer_detail": "Immediate burn management requires cooling with cool tap water (10–25 °C / 50–77 °F) to reduce pain and tissue injury. Ice water should not be used.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Burn",
                    "article_path": "Burn",
                    "fact_location": "Management section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Burn",
                        "performed with cool water 10–25 °C (50.0–77.0 °F) and not ice water",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Burn", "Cooling with tap water may help pain and decrease damage"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0023: Hypothermia treatment
    questions.append(
        {
            "id": "scene-0023",
            "question": "What are the recommended first-aid and medical rewarming procedures for individuals experiencing mild to moderate hypothermia?",
            "capability": "procedural",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "procedural", "first_aid", "emergency_medicine"],
            "expected_behavior": "answer",
            "answer": "Provide warm drinks, warm clothing, and physical activity for mild cases; use heating blankets and warmed intravenous fluids for moderate hypothermia.",
            "answer_detail": "Mild hypothermia management focuses on warm clothing, warm non-alcoholic drinks, and voluntary activity. Moderate hypothermia requires active rewarming techniques such as heating blankets and warmed intravenous fluids.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Hypothermia",
                    "article_path": "Hypothermia",
                    "fact_location": "Management section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Hypothermia",
                        "treatment of mild hypothermia involves warm drinks, warm clothing",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Hypothermia", "heating blankets and warmed intravenous fluids"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0024: Water purification
    questions.append(
        {
            "id": "scene-0024",
            "question": "What common physical and chemical treatment methods are utilized in household and municipal water purification to render raw water safe for consumption?",
            "capability": "procedural",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "procedural", "survival", "sanitation"],
            "expected_behavior": "answer",
            "answer": "Physical methods like boiling, filtration, sedimentation, and distillation, along with chemical disinfection using chlorine or chloramines.",
            "answer_detail": "Water purification procedures include physical processes such as filtration, sedimentation, distillation, and boiling, as well as chemical disinfection with chlorine to eliminate biological and chemical contaminants.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Water purification",
                    "article_path": "Water_purification",
                    "fact_location": "Processes and disinfection",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Water_purification", "Simple procedures such as boiling"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Water_purification", "filtration, sedimentation, and distillation"
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Water_purification",
                        "adding chlorine or chloramines as secondary disinfectants",
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0025: Composting procedure
    questions.append(
        {
            "id": "scene-0025",
            "question": "What fundamental material inputs and maintenance steps are required to manage a successful aerobic composting process?",
            "capability": "procedural",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "procedural", "gardening", "agriculture"],
            "expected_behavior": "answer",
            "answer": "Gathering a balance of green waste (nitrogen-rich) and brown waste (carbon-rich), maintaining moisture, and regularly turning the mixture for aeration.",
            "answer_detail": "Aerobic composting requires combining green waste (nitrogen-rich materials like leaves, grass, and food scraps) and brown waste (carbon-rich woody materials), adding water to maintain moisture, and providing proper aeration through regular turning.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Compost",
                    "article_path": "Compost",
                    "fact_location": "Process overview",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Compost",
                        "green waste (nitrogen-rich materials such as leaves, grass, and food scraps)",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Compost", "brown waste (woody materials rich in carbon"),
                    "source_index": 0,
                },
                {
                    "fact": check("Compost", "proper aeration by regularly turning the mixture"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0026: Choking first aid
    questions.append(
        {
            "id": "scene-0026",
            "question": "What physical first-aid interventions are standardly recommended to relieve acute foreign-body airway obstruction in a conscious choking adult?",
            "capability": "procedural",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "procedural", "first_aid", "emergency_medicine"],
            "expected_behavior": "answer",
            "answer": "Back blows (back slaps) and abdominal thrusts (the Heimlich maneuver).",
            "answer_detail": "To relieve severe choking in conscious individuals, first-aid protocols advise delivering back blows (back slaps) and performing abdominal thrusts (the Heimlich maneuver) to dislodge the obstructing object.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Choking",
                    "article_path": "Choking",
                    "fact_location": "Treatment section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Choking", "abdominal thrusts"), "source_index": 0},
                {"fact": check("Choking", "Heimlich maneuver"), "source_index": 0},
                {"fact": check("Choking", "Back blows (back slaps)"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0027: Bleeding control
    questions.append(
        {
            "id": "scene-0027",
            "question": "What is the primary initial first-aid action for external traumatic bleeding, and what intervention is used for severe limb hemorrhage if initial measures fail?",
            "capability": "procedural",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "procedural", "first_aid", "emergency_medicine"],
            "expected_behavior": "answer",
            "answer": "Applying direct pressure to the bleeding wound, and applying a tourniquet for severe limb injuries.",
            "answer_detail": "Traumatic external bleeding is primarily treated by applying direct pressure. For severely injured patients with uncontrolled limb bleeding, tourniquets are applied to stop blood loss.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Bleeding",
                    "article_path": "Bleeding",
                    "fact_location": "Treatment",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Bleeding", "treated by the application of direct pressure"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Bleeding", "tourniquets are helpful in preventing complications"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0028: Frostbite rewarming
    questions.append(
        {
            "id": "scene-0028",
            "question": "What is the recommended water temperature range and rewarming procedure for treating frostbite on affected extremities?",
            "capability": "procedural",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "procedural", "first_aid", "emergency_medicine"],
            "expected_behavior": "answer",
            "answer": "Immerse the affected extremity in a warm water bath maintained at 37–39 °C (98.6–102.2 °F).",
            "answer_detail": "Frostbite treatment requires rapid rewarming by immersion in warm water near body temperature, with wilderness medical guidelines recommending a temperature of 37–39 °C to reduce pain and tissue loss.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Frostbite",
                    "article_path": "Frostbite",
                    "fact_location": "Treatment section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Frostbite", "Treatment is by rewarming, immersion in warm water"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Frostbite", "temperature of 37–39 °C"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0029: Anaphylaxis treatment
    questions.append(
        {
            "id": "scene-0029",
            "question": "What is the first-line medication and precise anatomical injection site for the emergency treatment of acute anaphylaxis?",
            "capability": "procedural",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "procedural", "emergency_medicine", "pharmacology"],
            "expected_behavior": "answer",
            "answer": "Intramuscular injection of epinephrine into the mid-anterolateral thigh.",
            "answer_detail": "Guidelines recommend that an epinephrine solution be administered intramuscularly into the mid-anterolateral thigh as soon as anaphylaxis is suspected.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Anaphylaxis",
                    "article_path": "Anaphylaxis",
                    "fact_location": "Management / Epinephrine",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Anaphylaxis", "epinephrine solution be given intramuscularly"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Anaphylaxis",
                        "mid anterolateral thigh as soon as the diagnosis is suspected",
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0030: Lithium-ion battery safety
    questions.append(
        {
            "id": "scene-0030",
            "question": "What electrical and operational conditions must be avoided during lithium-ion battery usage to prevent thermal runaway and fire hazards?",
            "capability": "procedural",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "procedural", "electrical_engineering", "safety"],
            "expected_behavior": "answer",
            "answer": "Avoid short circuiting and overcharging to voltages higher than designed, which can cause overheating and thermal runaway.",
            "answer_detail": "Lithium-ion battery safety protocols require preventing internal and external short circuits and avoiding overcharge to voltages higher than designed, which can release heat and trigger thermal runaway.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Lithium-ion battery",
                    "article_path": "Lithium-ion_battery",
                    "fact_location": "Safety / Thermal runaway",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Lithium-ion_battery", "thermal runaway"), "source_index": 0},
                {
                    "fact": check(
                        "Lithium-ion_battery", "overcharge to voltages higher than designed"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Lithium-ion_battery", "short circuiting"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "procedural",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # =========================================================================
    # 4. COMPLEX EXPLANATION (scene-0031 .. scene-0040)
    # =========================================================================

    # scene-0031: Greenhouse effect
    questions.append(
        {
            "id": "scene-0031",
            "question": "How do atmospheric greenhouse gases warm Earth's surface through the differential absorption and re-emission of incoming solar radiation and outgoing thermal infrared radiation?",
            "capability": "complex_explanation",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "climate_science", "physics"],
            "expected_behavior": "answer",
            "answer": "Solar radiation passes largely unimpeded through greenhouse gases to warm Earth's surface, which in turn emits thermal infrared radiation that greenhouse gases absorb and re-radiate back toward the surface.",
            "answer_detail": "The atmosphere is largely transparent to incoming solar radiation, which warms the surface. The surface radiates thermal radiation (infrared) back out, where greenhouse gases absorb it and re-emit heat in all directions, warming the lower atmosphere and surface.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Greenhouse effect",
                    "article_path": "Greenhouse_effect",
                    "fact_location": "Mechanism section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Greenhouse_effect", "transparent to incoming solar radiation"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Greenhouse_effect", "heat-trapping gases in a planet's atmosphere"
                    ),
                    "source_index": 0,
                },
                {"fact": check("Greenhouse_effect", "thermal radiation"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0032: Plate tectonics
    questions.append(
        {
            "id": "scene-0032",
            "question": "How do mantle convection currents drive the movement of lithospheric plates, and what are the three major types of plate boundaries formed by their relative motions?",
            "capability": "complex_explanation",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "geology", "geophysics"],
            "expected_behavior": "answer",
            "answer": "Mantle convection currents drive the motion of rigid lithospheric plates floating on the ductile asthenosphere, forming convergent, divergent, and transform boundaries.",
            "answer_detail": "Lateral density variations and heat in the mantle create slow convection currents that move rigid lithospheric plates over the ductile asthenosphere. Their interactions create three main boundary types: convergent (colliding), divergent (spreading), and transform (sliding past).",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Plate tectonics",
                    "article_path": "Plate_tectonics",
                    "fact_location": "lede and Key principles",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Plate_tectonics",
                        "Earth's lithosphere comprises a number of large tectonic plates",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Plate_tectonics", "float across the ductile asthenosphere"),
                    "source_index": 0,
                },
                {"fact": check("Plate_tectonics", "convection currents"), "source_index": 0},
                {
                    "fact": check("Plate_tectonics", "convergent, divergent, or transform"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0033: Photosynthesis
    questions.append(
        {
            "id": "scene-0033",
            "question": "How do the light-dependent reactions in thylakoid membranes produce chemical energy, and how does the Calvin cycle subsequently utilize this energy to fix carbon dioxide?",
            "capability": "complex_explanation",
            "difficulty": "hard",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "biochemistry", "plant_biology"],
            "expected_behavior": "answer",
            "answer": "Light-dependent reactions in thylakoids capture light energy to synthesize ATP and NADPH, which the light-independent Calvin cycle then consumes to fix atmospheric carbon dioxide into sugars.",
            "answer_detail": "In thylakoid membranes, chlorophyll captures light to drive light-dependent reactions that produce ATP and NADPH. These energy carriers power the Calvin cycle (light-independent reactions) in the stroma to fix CO2 into carbohydrates.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Photosynthesis",
                    "article_path": "Photosynthesis",
                    "fact_location": "Overview and stages",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Photosynthesis", "light-dependent reactions"), "source_index": 0},
                {"fact": check("Photosynthesis", "thylakoids"), "source_index": 0},
                {
                    "fact": check("Photosynthesis", "synthesize adenosine triphosphate (ATP)"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Photosynthesis", "light-independent reactions called the Calvin cycle"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0034: Ozone depletion
    questions.append(
        {
            "id": "scene-0034",
            "question": "What is the multi-step chemical mechanism by which atmospheric chlorofluorocarbons (CFCs) release chlorine radicals that catalytically destroy stratospheric ozone?",
            "capability": "complex_explanation",
            "difficulty": "hard",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "chemistry", "atmospheric_science"],
            "expected_behavior": "answer",
            "answer": "CFCs migrate to the stratosphere where ultraviolet light breaks them down to release reactive chlorine radicals, which catalytically break down ozone molecules into oxygen through recurring reaction cycles.",
            "answer_detail": "CFCs are transported into the stratosphere where solar ultraviolet radiation photolyzes them to liberate chlorine radicals (Cl·). These radicals engage in catalytic cycles that repeatedly destroy ozone molecules without being consumed themselves.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Ozone depletion",
                    "article_path": "Ozone_depletion",
                    "fact_location": "Mechanism and catalytic cycles",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Ozone_depletion", "chlorofluorocarbons (CFCs)"), "source_index": 0},
                {
                    "fact": check("Ozone_depletion", "transported into the stratosphere"),
                    "source_index": 0,
                },
                {"fact": check("Ozone_depletion", "chlorine radical (Cl·)"), "source_index": 0},
                {"fact": check("Ozone_depletion", "catalytic cycles"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0035: Tides
    questions.append(
        {
            "id": "scene-0035",
            "question": "How do the differential gravitational forces of the Moon and Sun combined with orbital inertia create oceanic tides, and what produces spring versus neap tides?",
            "capability": "complex_explanation",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "oceanography", "astronomy"],
            "expected_behavior": "answer",
            "answer": "Differential gravitational pull across Earth creates high and low tidal cycles; alignment of the Moon and Sun produces maximum tidal ranges (spring tides), while perpendicular alignment produces minimal ranges (neap tides).",
            "answer_detail": "Tides are the rise and fall of sea level caused by differential gravitational forces exerted by the Moon and Sun combined with inertial effects. When the Sun and Moon align, their gravitational effects combine to form higher-range spring tides; when at right angles, they produce lower-range neap tides.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Tide",
                    "article_path": "Tide",
                    "fact_location": "lede and tidal variations",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Tide",
                        "differential gravitational forces exerted primarily by the Moon and the Sun",
                    ),
                    "source_index": 0,
                },
                {"fact": check("Tide", "spring tides"), "source_index": 0},
                {"fact": check("Tide", "neap tides"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0036: CRISPR gene editing
    questions.append(
        {
            "id": "scene-0036",
            "question": "How does the CRISPR-Cas9 system identify a specific target DNA sequence, generate double-strand breaks, and enable targeted genomic modification?",
            "capability": "complex_explanation",
            "difficulty": "hard",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "biotechnology", "genetics"],
            "expected_behavior": "answer",
            "answer": "Synthetic guide RNA directs the Cas9 nuclease to target DNA adjacent to a PAM motif, where Cas9 introduces a double-strand break that is repaired by cellular repair mechanisms like non-homologous end joining.",
            "answer_detail": "In CRISPR gene editing, a Cas9 endonuclease complexed with a synthetic guide RNA (gRNA) recognizes a complementary target DNA sequence adjacent to a protospacer adjacent motif (PAM) and generates a double-strand break. Cellular repair via non-homologous end joining (NHEJ) or homology-directed repair introduces targeted edits.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "CRISPR gene editing",
                    "article_path": "CRISPR_gene_editing",
                    "fact_location": "Mechanism section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "CRISPR_gene_editing", "Cas9 nuclease complexed with a synthetic guide RNA"
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("CRISPR_gene_editing", "protospacer adjacent motif (PAM)"),
                    "source_index": 0,
                },
                {
                    "fact": check("CRISPR_gene_editing", "non-homologous end joining"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0037: Superconductivity
    questions.append(
        {
            "id": "scene-0037",
            "question": "According to BCS theory, how do electron-phonon interactions lead to the formation of Cooper pairs, resulting in zero electrical resistance and the Meissner effect below the critical temperature?",
            "capability": "complex_explanation",
            "difficulty": "hard",
            "slice": "core",
            "tags": [
                "scenario",
                "complex_explanation",
                "condensed_matter_physics",
                "quantum_mechanics",
            ],
            "expected_behavior": "answer",
            "answer": "Electrons interact through lattice vibrations to form Cooper pairs that condense into a superfluid condensate, dropping electrical resistance to zero below the critical temperature and expelling magnetic fields (the Meissner effect).",
            "answer_detail": "Under BCS theory, electrons form bound pairs called Cooper pairs through attractive electron-phonon lattice interactions. Below the critical temperature, this collective state acts as a superfluid carrying current without resistance and completely expelling magnetic flux lines via the Meissner effect.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Superconductivity",
                    "article_path": "Superconductivity",
                    "fact_location": "BCS theory and Meissner effect",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Superconductivity", "superfluid of Cooper pairs"),
                    "source_index": 0,
                },
                {"fact": check("Superconductivity", "critical temperature"), "source_index": 0},
                {"fact": check("Superconductivity", "Meissner effect"), "source_index": 0},
                {"fact": check("Superconductivity", "BCS theory"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0038: Inflation mechanisms
    questions.append(
        {
            "id": "scene-0038",
            "question": "How do demand-pull and cost-push mechanisms differ in their macroeconomic causes of price level inflation?",
            "capability": "complex_explanation",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "economics", "macroeconomics"],
            "expected_behavior": "answer",
            "answer": "Demand-pull inflation is caused by increases in aggregate demand exceeding productive capacity, whereas cost-push inflation is caused by negative supply shocks or drops in aggregate supply that increase production costs.",
            "answer_detail": "Inflation represents a reduction in purchasing power. Demand-pull inflation occurs when aggregate demand surges and outpaces supply, while cost-push inflation is triggered by supply shocks or increased costs of production that reduce aggregate supply.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Inflation",
                    "article_path": "Inflation",
                    "fact_location": "Causes / Demand-pull and Cost-push",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Inflation", "purchasing power of money"), "source_index": 0},
                {"fact": check("Inflation", "demand-pull inflation"), "source_index": 0},
                {
                    "fact": check(
                        "Inflation", "Cost-push inflation is caused by a drop in aggregate supply"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0039: Action potential
    questions.append(
        {
            "id": "scene-0039",
            "question": "What sequence of voltage-gated ion channel openings and ion flows produces the rapid depolarization and repolarization phases of a neuronal action potential?",
            "capability": "complex_explanation",
            "difficulty": "hard",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "neuroscience", "physiology"],
            "expected_behavior": "answer",
            "answer": "Reaching a threshold voltage opens voltage-gated sodium channels causing rapid inward Na+ influx (depolarization), followed by sodium channel inactivation and opening of voltage-gated potassium channels causing outward K+ efflux (repolarization).",
            "answer_detail": "When membrane potential reaches a threshold, voltage-gated sodium channels open rapidly, allowing an inward flow of sodium ions that causes depolarization. Subsequently, sodium channels inactivate and voltage-gated potassium channels activate, allowing potassium ions to exit the cell to drive repolarization.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Action potential",
                    "article_path": "Action_potential",
                    "fact_location": "Underlying mechanism",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Action_potential", 'threshold voltage, "depolarising" the membrane'
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Action_potential", "inward flow of sodium ions"),
                    "source_index": 0,
                },
                {
                    "fact": check("Action_potential", "Potassium channels are then activated"),
                    "source_index": 0,
                },
                {"fact": check("Action_potential", "repolarization"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0040: Doppler effect
    questions.append(
        {
            "id": "scene-0040",
            "question": "How does relative motion between a wave source and an observer cause wave compression and frequency shifts (such as blueshift and redshift)?",
            "capability": "complex_explanation",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "complex_explanation", "physics", "acoustics_and_optics"],
            "expected_behavior": "answer",
            "answer": "Motion of the source towards an observer compresses successive wave crests, increasing the perceived frequency (blueshift), while motion away stretches wave crests, decreasing the perceived frequency (redshift).",
            "answer_detail": "The Doppler effect is the change in frequency of a wave relative to an observer moving relative to the wave source. Waves emitted by an approaching source are bunched together (shorter wavelength, higher frequency/blueshift), while waves from a receding source are stretched out (longer wavelength, lower frequency/redshift).",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Doppler effect",
                    "article_path": "Doppler_effect",
                    "fact_location": "lede and Mechanism",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Doppler_effect",
                        "change in the frequency or, equivalently, the period of a wave in relation to an observer",
                    ),
                    "source_index": 0,
                },
                {"fact": check("Doppler_effect", "blueshift or redshift"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "complex_explanation",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # =========================================================================
    # 5. ADVERSARIAL ABSTENTION (scene-0041 .. scene-0050)
    # =========================================================================

    # scene-0041: Great Wall visible from Moon
    questions.append(
        {
            "id": "scene-0041",
            "question": "What color was the Great Wall of China painted during the Ming Dynasty so that astronauts on the lunar surface could view it with the naked eye?",
            "capability": "adversarial_abstention",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "myth_busting", "space_myths"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. The Great Wall of China cannot be seen from the Moon with the naked eye, and it was never painted for astronauts.",
            "answer_detail": "The Great Wall of China cannot be seen from the Moon; even from low Earth orbit, it is generally not visible to the unaided eye. The claim that it was constructed or painted to be visible from the Moon is an urban myth.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Great Wall of China",
                    "article_path": "Great_Wall_of_China",
                    "fact_location": "From the Moon section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Great_Wall_of_China",
                        "The Great Wall of China cannot be seen by the naked human eye from the Moon",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Great_Wall_of_China", "Even though the myth has been thoroughly debunked"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0042: Lemming mass suicide
    questions.append(
        {
            "id": "scene-0042",
            "question": "During which phase of the lunar cycle do Arctic lemmings gather to commit mass suicide by jumping off cliffs into the ocean?",
            "capability": "adversarial_abstention",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "biology_myths", "zoology"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Lemmings do not commit mass suicide by jumping off cliffs; this is a popular misconception largely popularized by a staged 1958 Disney film.",
            "answer_detail": "The idea that lemmings exhibit herd mentality and jump off cliffs to commit mass suicide is a myth. The footage popularizing this behavior was staged in the 1958 Walt Disney documentary White Wilderness.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Lemming",
                    "article_path": "Lemming",
                    "fact_location": "Misconceptions / lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Lemming",
                        "A longstanding myth claims that they exhibit herd mentality and jump off cliffs, committing mass suicide.",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Lemming", "Walt Disney documentary White Wilderness in 1958"),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0043: Phlogiston theory oxygen absorption
    questions.append(
        {
            "id": "scene-0043",
            "question": "In 18th-century phlogiston theory, how many grams of atmospheric oxygen gas are absorbed by a combustible metal during calcination?",
            "capability": "adversarial_abstention",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "history_of_science", "chemistry"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Phlogiston theory posited that combustible substances release phlogiston upon burning, not that they absorb oxygen.",
            "answer_detail": "Phlogiston theory was a superseded chemical theory asserting that combustible bodies contain a fire-like element called phlogiston which is released during combustion and calcination, rather than absorbing atmospheric oxygen.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Phlogiston theory",
                    "article_path": "Phlogiston_theory",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Phlogiston_theory",
                        "contained within combustible bodies and released during combustion",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Phlogiston_theory", "experiments by Antoine Lavoisier in the 1770s"
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0044: Spontaneous generation of maggots
    questions.append(
        {
            "id": "scene-0044",
            "question": "According to modern cell biology, what ambient temperature is required for sterile dead flesh to spontaneously generate live maggots without prior fly egg contact?",
            "capability": "adversarial_abstention",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "biology", "history_of_science"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Modern biology disproved spontaneous generation; maggots only develop on flesh if adult flies deposit eggs on it.",
            "answer_detail": "Spontaneous generation—the idea that living organisms like maggots could spontaneously arise from dead flesh—is a superseded scientific theory disproven by experiments by Francesco Redi and Louis Pasteur.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Spontaneous generation",
                    "article_path": "Spontaneous_generation",
                    "fact_location": "lede and Redi experiments",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Spontaneous_generation",
                        "Spontaneous generation is a superseded scientific theory that held that living creatures could arise from non-living matter",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check("Spontaneous_generation", "maggots could arise from dead flesh"),
                    "source_index": 0,
                },
                {"fact": check("Spontaneous_generation", "Francesco Redi"), "source_index": 0},
                {"fact": check("Spontaneous_generation", "Louis Pasteur"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0045: Bermuda Triangle magnetic vortex
    questions.append(
        {
            "id": "scene-0045",
            "question": "What frequency of ultrasonic radiation emitted by the underwater magnetic vortex in the Bermuda Triangle causes aircraft compasses to spin out of control?",
            "capability": "adversarial_abstention",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "pseudoscience", "navigation"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Maritime and aviation authorities (including the US Coast Guard and Lloyd's of London) do not recognize magnetic vortices or anomalous disappearance rates in the Bermuda Triangle.",
            "answer_detail": "The notion of a magnetic vortex emitting ultrasonic waves causing mysterious disappearances in the Bermuda Triangle is unsupported pseudoscience. Official records from the United States Coast Guard and Lloyd's of London confirm that the area does not experience an unusually high rate of maritime or aviation losses.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Bermuda Triangle",
                    "article_path": "Bermuda_Triangle",
                    "fact_location": "Criticisms and records",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Bermuda_Triangle",
                        "United States Coast Guard records confirm their conclusion",
                    ),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Bermuda_Triangle",
                        "Lloyd's determined that large numbers of ships had not sunk there",
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0046: Loch Ness Monster specimen
    questions.append(
        {
            "id": "scene-0046",
            "question": "Which species of Jurassic marine reptile was officially cataloged by Scottish zoologists following the live capture of the Loch Ness Monster in 1934?",
            "capability": "adversarial_abstention",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "cryptozoology", "hoaxes"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. No live specimen of the Loch Ness Monster was ever captured or cataloged, and the famous 1934 'Surgeon's Photograph' was an established hoax.",
            "answer_detail": "The Loch Ness Monster is a cryptozoological creature with no scientific evidence supporting its existence. No specimen was ever captured or scientifically cataloged, and the famous 1934 photograph was revealed to be a hoax.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Loch Ness Monster",
                    "article_path": "Loch_Ness_Monster",
                    "fact_location": "lede and Surgeon's photograph",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Loch_Ness_Monster", "cryptozoology"), "source_index": 0},
                {
                    "fact": check(
                        "Loch_Ness_Monster",
                        '"surgeon\'s photograph" of 1934, now known to have been a hoax',
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0047: Rabies cure with vitamin C
    questions.append(
        {
            "id": "scene-0047",
            "question": "What daily oral dosage of mega-dose vitamin C has been clinically proven to reverse encephalitis and cure active rabies symptoms in humans?",
            "capability": "adversarial_abstention",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "medical_myths", "infectious_disease"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Rabies is virtually 100% fatal once clinical neurological symptoms appear, and vitamin C cannot cure or reverse symptomatic rabies.",
            "answer_detail": "Rabies causes acute and severe viral encephalitis and is almost 100% fatal once symptoms manifest. There is no clinically proven cure for symptomatic rabies, including vitamin C; prevention relies entirely on vaccination and timely post-exposure prophylaxis before symptom onset.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Rabies",
                    "article_path": "Rabies",
                    "fact_location": "lede and prognosis",
                    "required": True,
                }
            ],
            "sub_facts": [
                {"fact": check("Rabies", "acute and severe encephalitis"), "source_index": 0},
                {"fact": check("Rabies", "~100% fatal after onset of symptoms"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0048: Atlantis excavation by Napoleon
    questions.append(
        {
            "id": "scene-0048",
            "question": "Which French archaeological brigade led by Napoleon Bonaparte successfully excavated the submerged bronze palaces of Atlantis in the central Atlantic Ocean in 1798?",
            "capability": "adversarial_abstention",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "historical_fallacy", "mythology"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Atlantis is an allegorical myth introduced by Plato, and Napoleon never discovered or excavated submerged palaces in the Atlantic.",
            "answer_detail": "Atlantis is a fictional island mentioned in Plato's dialogues Timaeus and Critias as an allegory on the hubris of nations. No physical submerged palaces of Atlantis exist, and Napoleon Bonaparte never conducted archaeological excavations of Atlantis.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Atlantis",
                    "article_path": "Atlantis",
                    "fact_location": "lede",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check(
                        "Atlantis",
                        "fictional island mentioned in Plato's works Timaeus and Critias",
                    ),
                    "source_index": 0,
                },
                {"fact": check("Atlantis", "allegory on the hubris of nations"), "source_index": 0},
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0049: Wardenclyffe transatlantic wireless power transmission
    questions.append(
        {
            "id": "scene-0049",
            "question": "How many megawatts of commercial electrical power did Nikola Tesla successfully transmit wirelessly across the Atlantic Ocean from his Wardenclyffe Tower to England in 1904?",
            "capability": "adversarial_abstention",
            "difficulty": "medium",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "historical_fallacy", "biography"],
            "expected_behavior": "abstain",
            "answer": "The premise is false. Tesla's Wardenclyffe Tower was never completed or fully operational, and it never transmitted transatlantic wireless power.",
            "answer_detail": "Nikola Tesla's Wardenclyffe Tower was an ambitious but unfinished experimental project intended for wireless communication and power transmission. It was never completed or operational due to financial difficulties and was never used to transmit commercial power across the Atlantic.",
            "sources": [
                {
                    "zim": "wikipedia_en_top_nopic_2026-06.zim",
                    "article_title": "Nikola Tesla",
                    "article_path": "Nikola_Tesla",
                    "fact_location": "Wardenclyffe Tower section",
                    "required": True,
                }
            ],
            "sub_facts": [
                {
                    "fact": check("Nikola_Tesla", "unfinished Wardenclyffe Tower project"),
                    "source_index": 0,
                },
                {
                    "fact": check(
                        "Nikola_Tesla",
                        "intercontinental wireless communication and power transmitter",
                    ),
                    "source_index": 0,
                },
            ],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    # scene-0050: Out-of-corpus pure fiction
    questions.append(
        {
            "id": "scene-0050",
            "question": "What is the atomic mass and crystal lattice structure of the synthetic superheavy element 'Veridanium-124' discovered at the Neo-Kyoto Hadron Facility in 2031?",
            "capability": "adversarial_abstention",
            "difficulty": "easy",
            "slice": "core",
            "tags": ["scenario", "adversarial_abstention", "out_of_corpus", "pure_fiction"],
            "expected_behavior": "abstain",
            "answer": "The query refers to a fictional element ('Veridanium-124') and facility ('Neo-Kyoto Hadron Facility') not present in the scientific record or knowledge base.",
            "answer_detail": "No element named Veridanium or research facility named Neo-Kyoto Hadron Facility exists in the Wikipedia corpus or real-world scientific literature.",
            "sources": [],
            "sub_facts": [],
            "provenance": {
                "authored_by": "bench_authoring_agent",
                "authored_at": "2026-08-15",
                "scenario_type": "adversarial_abstention",
            },
            "closed_book": {},
            "oracle": {},
            "status": "active",
        }
    )

    return questions


def main():
    questions = build_questions()
    print(f"Authored {len(questions)} scenario questions.")
    out_dir = Path("data/bench_logs/authoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "scenario_50.json"
    out_file.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote to {out_file}")


if __name__ == "__main__":
    main()

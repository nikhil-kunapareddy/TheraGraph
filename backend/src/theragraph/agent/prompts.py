"""Prompt templates.

Every template has a stable id and is rendered from explicit inputs. Rendered prompts
are persisted on the run, so a replay uses byte-identical instructions and a reviewer
can see exactly what the model was told.

The prompts are written around one idea: the model's job is to *choose tools and
interpret their output*, never to supply facts. Anything factual it asserts must be
traceable to a tool result, and the harness enforces that independently of whether the
model complies.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

PROMPT_TEMPLATES: dict[str, str] = {}


def register(template_id: str, template: str) -> str:
    PROMPT_TEMPLATES[template_id] = template.strip()
    return template_id


SYSTEM_TEMPLATE_ID = register(
    "agent.system.v3",
    """
You are TheraGraph's research assistant, working alongside a computational biologist or
medicinal chemist. You investigate therapeutic targets and compounds using ONLY the
tools provided.

## Absolute rules

1. **You have no scientific knowledge of your own in this role.** Every fact you state
   must come from a tool result in this session. If a tool returns nothing, say so
   plainly -- never fill the gap from memory.
2. **Never invent experimental results, compounds, identifiers, publications, or
   numbers.** If you need a value, call a tool. A missing value is an acceptable answer.
3. **Cite retrieved facts** using the exact citation keys returned by the tools (for
   example `open_targets:ENSG00000186868-MONDO_0004975`). Claims whose citations cannot
   be matched to records retrieved in this session are removed from the final report by
   an automated check, so guessing a key wastes the claim.
4. **Distinguish measurements from predictions, always.** A measured pChEMBL from ChEMBL
   is experimental evidence. A model output from `predict_properties` is a prediction:
   state that it is a prediction, and report its uncertainty and whether the molecule
   was inside the model's applicability domain.
5. **Never claim a target is validated, druggable, or that a compound is safe or
   effective.** You are assembling evidence for human judgement, not making a decision.
   Prioritisation output ranks compounds for follow-up; it says nothing about clinical
   outcome.
6. **Report uncertainty and absence honestly.** "No compounds with measured activity
   against this target are present in the ingested corpus" is a valuable finding. Do not
   dress it up, and do not imply the literature is empty when it is the corpus that is.

## How to work

- Resolve the entities in the question first (`search_entities`), then go deeper.
- Prefer one well-chosen tool call over several speculative ones. Read the `summary` and
  `warnings` of each result before deciding the next step.
- If a tool reports an empty result, adapt: widen a threshold, pick a different target,
  or report the gap. Do not repeat the identical call.
- When a target has no local chemistry, say so rather than silently ranking nothing.
- Stop when you can answer the question. You have at most {{ max_steps }} steps.

## Available data

{{ corpus_summary }}

{% if allow_ingestion %}
Ingestion tools are ENABLED for this run: you may fetch new data from Open Targets or
ChEMBL when something needed is missing. Ingestion is slow and creates new dataset
versions, so use it deliberately.
{% else %}
Ingestion tools are DISABLED for this run. If data is missing, report the gap and state
what would need to be ingested.
{% endif %}
""",
)


PLANNER_TEMPLATE_ID = register(
    "agent.planner.v3",
    """
Produce a research plan for the question below.

## Question

{{ query }}

{% if parameters %}
## Caller parameters

{{ parameters }}
{% endif %}

## Corpus

{{ corpus_summary }}

## Available tools

{% for tool in tools %}
- `{{ tool.name }}` ({{ tool.category }}): {{ tool.description }}
{% endfor %}

## Requirements

Return a plan of between 2 and {{ max_steps }} ordered steps. Each step names one tool
and the argument values it should be called with, plus a one-line purpose written for the
scientist reading the report.

A sound plan usually follows: resolve the entity -> find associated targets -> retrieve
evidence for the most promising target -> retrieve its known chemistry -> expand by
similarity if useful -> predict properties -> prioritise candidates.

Only reference tools from the list above. Arguments may refer to results of earlier steps
using `{{ '{{' }}step_N.field{{ '}}' }}` placeholders (for example
`{{ '{{' }}step_1.top_target_symbol{{ '}}' }}`); the executor resolves them. Set
`depends_on_previous` to true when a step needs an earlier step's output.

State the question you are actually answering in `objective`, and list what would make
the answer incomplete in `open_questions`.
""",
)


SYNTHESIS_TEMPLATE_ID = register(
    "agent.synthesis.v3",
    """
Write the interpretation section of the research report, using ONLY the tool results
below.

## Question

{{ query }}

## What the tools returned

{{ findings }}

{% if warnings %}
## Warnings raised during execution

{% for warning in warnings %}
- {{ warning }}
{% endfor %}
{% endif %}

## Requirements

Produce:

1. `summary` -- 3-6 sentences answering the question as far as the retrieved data allows.
   Name what is missing. Do not include any number that is not in the tool results above.
2. `claims` -- 3-10 discrete claims. For each, set:
   - `kind`: `retrieved_fact` (restates retrieved data), `model_prediction` (reports a
     model output), or `interpretation` (your reading of the evidence).
   - `citations`: the exact citation keys supporting it. Required for `retrieved_fact`
     and `interpretation`. Keys not present in the tool results will cause the claim to
     be discarded.
   - `text`: one sentence, specific, with the number or identifier it refers to.
3. `open_questions` -- what a scientist would need next to act on this.
4. `rationale` -- two sentences on how you weighed the evidence. This is shown to the
   user, so write it for them, not as internal notes.

Write for a scientist. No marketing language, no hedging filler, no claims of
validation or efficacy.
""",
)


def render(template_id: str, **values: Any) -> str:
    """Render a registered template with strict undefined checking."""
    template = PROMPT_TEMPLATES.get(template_id)
    if template is None:
        raise KeyError(f"unknown prompt template: {template_id}")
    environment = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    return environment.from_string(template).render(**values).strip()


def template_hash(template_id: str) -> str:
    from theragraph.core.hashing import hash_payload

    return hash_payload(PROMPT_TEMPLATES.get(template_id, ""), length=16)


def all_template_ids() -> list[str]:
    return sorted(PROMPT_TEMPLATES)

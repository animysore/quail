"""Generate and validate named string fields for document rows."""

import json
import time
from dataclasses import dataclass

import pyarrow as pa

from quail.execution.runner import NodeMetrics, NodeResult
from quail.progress import answer_sink

EXTRACT_BATCH_ROWS = 128


@dataclass(frozen=True)
class GenerationResult:
    """One generated response and the tokens actually evaluated."""

    text: str
    finish_reason: str
    prompt_tokens: int
    cached_tokens: int = 0
    output_tokens: int = 0


def extraction_schema(fields: tuple[str, ...]) -> dict:
    """Return the JSON schema enforced during generation."""
    return {
        "type": "object",
        "properties": {name: {"type": ["string", "null"]} for name in fields},
        "required": list(fields),
        "additionalProperties": False,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def extraction_result(output: GenerationResult, fields: tuple[str, ...]) -> dict:
    """Separate valid missing values from failed or truncated generation."""
    if output.finish_reason == "incomplete":
        return {"response": None, "error": "Incomplete extraction response"}
    if output.finish_reason == "context":
        return {"response": None, "error": "Extraction exceeded its context limit"}
    if output.finish_reason != "stop":
        return {"response": None, "error": "Extraction exceeded its output limit"
                if output.finish_reason == "length"
                else "Extraction generation did not finish"}
    try:
        value = json.loads(output.text, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != set(fields):
            raise ValueError("wrong extraction fields")
        if any(item is not None and not isinstance(item, str)
               for item in value.values()):
            raise ValueError("extraction values must be strings or null")
    except (ValueError, TypeError):
        return {"response": None, "error": "Invalid extraction JSON"}
    return {"response": value, "error": None}


def extract_rows(node, inputs, document_texts, client) -> NodeResult:
    """Append an extraction column, preserving row indices and earlier columns."""
    if len(inputs) != 1:
        raise ValueError("AI.EXTRACT needs one input relation")
    source = next(iter(inputs.values()))
    table = source if isinstance(source, pa.Table) else pa.table({
        node.alias: pa.array(source, type=pa.int32()),
    })
    if node.name in table.column_names:
        raise ValueError("AI.EXTRACT output already exists in its input")
    texts = document_texts[node.alias]
    dtype = pa.struct([
        pa.field("response", pa.struct([
            pa.field(name, pa.string()) for name in node.fields
        ])),
        pa.field("error", pa.string()),
    ])
    started = time.perf_counter()
    chunks = []
    requests = prompt_tokens = cached_tokens = output_tokens = errors = 0
    rejected_requests = 0
    schema = extraction_schema(node.fields)
    for batch in table.to_batches(max_chunksize=EXTRACT_BATCH_ROWS):
        indices = batch.column(node.alias).to_pylist()
        positions, prompts = [], []
        values = [None] * len(indices)
        for position, index in enumerate(indices):
            text = texts[index]
            text = text.as_py() if hasattr(text, "as_py") else text
            if text is None:
                continue
            if not isinstance(text, str):
                raise TypeError("AI.EXTRACT input must be text")
            positions.append(position)
            prompts.append(node.preamble_text + text + node.tail_text)
        outputs = client.generate_structured(
            prompts, schema, node.max_output_tokens
        ) if prompts else []
        if len(outputs) != len(positions):
            raise RuntimeError("AI.EXTRACT returned the wrong number of rows")
        for position, output in zip(positions, outputs):
            value = extraction_result(output, node.fields)
            values[position] = value
            errors += value["error"] is not None
            rejected = output.finish_reason == "context"
            rejected_requests += rejected
            requests += not rejected
            prompt_tokens += output.prompt_tokens
            cached_tokens += output.cached_tokens
            output_tokens += output.output_tokens
        chunks.append(pa.array(values, type=dtype))
        sink = answer_sink()
        if sink is not None:
            sink({
                "kind": "extract", "node": node.node_id, "output": node.name,
                "alias": node.alias, "rows": indices, "extractions": values,
            })
    result = table.append_column(node.name, pa.chunked_array(chunks, type=dtype))
    return NodeResult(
        {"rows": result},
        NodeMetrics(
            wall_s=time.perf_counter() - started,
            input_rows=len(table), output_rows=len(table),
            evaluated_documents=requests,
            fresh_tokens=prompt_tokens - cached_tokens, cached_tokens=cached_tokens,
            extension={
                "output": node.name, "fields": list(node.fields),
                "requests": requests, "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens, "errors": errors,
                "rejected_requests": rejected_requests,
            },
        ),
    )

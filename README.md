# Quail

Quail (QUery Aware Inference Layer) is an open-source, extensible
execution engine for AI-SQL, being developed at
[Full Stack Data Lab](https://fsdatalab.github.io/) at CMU.

AI-SQL is a variant of SQL with operators that let you write logic
in natural language for an LLM to evaluate on every row.

```sql
SELECT r.id
FROM reviews r
WHERE AI.IF(PROMPT('Does this review discuss the ending?\n\n{0}', r.body))
```

[Documentation](https://fsdatalab.github.io/quail) |
[Quickstart](https://fsdatalab.github.io/quail/docs/user-guide/quickstart) |
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## Install

Install [quail-engine from PyPI](https://pypi.org/project/quail-engine/).
The package is named `quail-engine`; in Python, import `quail`.

```bash
uv pip install quail-engine
```

Requires Python 3.12 and a CUDA GPU.

## Example

```python
import pyarrow as pa
import quail

reviews = pa.table({
    "id": ["r1", "r2", "r3"],
    "body": [
        "A beautiful film with outstanding performances.",
        "Terrible pacing and a nonsensical plot.",
        "The cinematography was stunning, though the story dragged.",
    ],
})

config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
with quail.Session(config=config) as session:
    session.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
    result = session.sql("""
        SELECT r.id
        FROM reviews r
        WHERE AI.IF(PROMPT(
            'Does this review mention a positive aspect of the movie?\n\n{0}',
            r.body))
    """, dialect="bq").collect()
    print(result)
```

## Quail Server

Quail Server is an optional HTTP server that runs on the machine with
the GPU. You start it once, send queries to it with `endpoint`, and
the query keeps running after the client disconnects. `submit()`
returns once the server has saved the record, and `get_run` reads
that record later.

```bash
uv pip install "quail-engine[server]"
quail-server
```

```python
import pyarrow as pa
import quail

reviews = pa.table({
    "id": ["r1", "r2"],
    "body": ["The ending was excellent.", "I liked the soundtrack."],
})
sql = """
    SELECT r.id
    FROM reviews r
    WHERE AI.IF(PROMPT('Does this discuss the ending? {0}', r.body))
"""
config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")

with quail.Session(
    config=config,
    endpoint="http://127.0.0.1:8642",
) as session:
    session.register(
        "reviews",
        quail.DocumentProvider.from_table(reviews, id_col="id"),
    )
    run = session.sql(sql, dialect="bq").submit()
    for status in run.watch():
        print(status.phase["message"])
    table = run.result().collect()
    print(table)
```

See [Quail Server](https://fsdatalab.github.io/quail/docs/user-guide/server).

## Supported operators

Quail currently supports AI-powered filters, joins, and
`EXISTS` / `NOT EXISTS`. `AI.EXTRACT` returns named string fields
through the vLLM backends with generative Qwen3 models.
We are actively adding more operators (`AI.CLASSIFY`, `AI.MAP`).

We support two AI-SQL dialects:
[Snowflake `AI_FILTER`](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
and
[BigQuery `AI.IF`](https://cloud.google.com/blog/products/data-analytics/sql-reimagined-for-the-ai-era-with-bigquery-ai-functions),
plus a Python builder API.

## Supported models and GPUs

| Model | Device |
| --- | --- |
| Qwen3 4B fp8 | NVIDIA H100 SXM |
| Qwen3 32B fp8 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| DiffusionGemma 26B-A4B fp8 | NVIDIA H100 SXM |

1, 2, 4, or 8 GPUs per query. We are actively adding more models
and hardware.

## Development

```bash
uv run ruff check quail tests experiments tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```

## Contributing

See the [contributing guide](https://fsdatalab.github.io/quail/docs/contributing)
for how to propose and submit changes.

# AGENTS.md

This document provides guidance for AI coding agents (GitHub Copilot, Claude, GPT, etc.) working on the **mergekit** repository.

---

## Repository Overview

`mergekit` is a toolkit for merging pre-trained language models using out-of-core tensor operations. It supports CPU and GPU execution and provides a rich set of merge algorithms. Key packages:

| Package | Purpose |
|---|---|
| `mergekit` | Core merge engine (config, planning, execution, graph) |
| `mergekit.merge_methods` | Individual merge algorithm implementations |
| `mergekit.io` | Lazy tensor I/O and shard management |
| `mergekit.architecture` | Model architecture detection and weight metadata |
| `mergekit.tokenizer` | Tokenizer handling and vocabulary alignment |
| `mergekit.moe` | Mixture-of-experts model creation |
| `mergekit.evo` | Evolutionary merge method support |
| `mergekit.tokensurgeon` | Tokenizer transplantation between models |
| `mergekit.scripts` | CLI entry points and the MCP server |

---

## Development Setup

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e ".[dev]"
pre-commit install
```

For the MCP server:

```sh
pip install -e ".[mcp]"
```

---

## Running Tests

```sh
pytest tests/
```

Tests live in `tests/`. They use `pytest` and do **not** require GPU or real model weights.

---

## Code Style

- Formatter: **black** (`line-length = 88`)
- Import order: **isort** (`profile = "black"`)
- Pre-commit hooks enforce both; run `pre-commit run --all-files` before submitting a PR.
- License headers follow the form `# Copyright (C) <year> Arcee AI` / `# SPDX-License-Identifier: LGPL-3.0-only`.

---

## Adding a New Merge Method

1. Create `mergekit/merge_methods/<method_name>.py`.
2. Subclass `MergeMethod` from `mergekit.merge_methods.base`.
3. Implement `name()`, `parameters()`, `tensor_parameters()`, and `make_task()`.
4. Register the method in `mergekit/merge_methods/registry.py` by appending an instance to `STATIC_MERGE_METHODS`.
5. Add documentation to `docs/merge_methods.md` and update the method table in `README.md`.

---

## MCP Server

The MCP server (`mergekit/scripts/mcp_server.py`) exposes mergekit's functionality via the [Model Context Protocol](https://modelcontextprotocol.io/). It is built with `mcp.server.fastmcp.FastMCP` and uses the `mcp` optional-dependency group.

### Available Tools

| Tool | Description |
|---|---|
| `list_merge_methods` | List all registered merge methods with parameters |
| `get_merge_method_info` | Get details for a single merge method by name |
| `validate_merge_config` | Validate a YAML merge configuration without running it |
| `run_merge_from_config` | Execute a merge and write the model to a given path |
| `generate_merge_card` | Generate a Hugging Face model card for a configuration |

### Adding a New MCP Tool

1. Open `mergekit/scripts/mcp_server.py`.
2. Define an `async` or synchronous function and decorate it with `@mcp.tool(description="...")`.
3. Use Python type annotations for all parameters — FastMCP derives the JSON schema automatically.
4. Return a plain string (JSON-encoded where structured data is needed).
5. Document the new tool in this file and in `README.md`.

---

## Configuration Format

Merge configurations are YAML documents. The authoritative schema is `mergekit.config.MergeConfiguration` (a Pydantic model). All field names, types, and validation rules are defined there. When in doubt, read the model — it is the single source of truth.

---

## Important Design Notes

- **Out-of-core execution**: tensors are loaded lazily via `mergekit.io`. Never load entire models into memory directly.
- **Graph-based execution**: merge plans are compiled into a dependency graph (`mergekit.graph`). New operations must fit this model.
- **Frozen options**: `MergeOptions` is a frozen Pydantic model. Do not add mutable state to it.
- **Pydantic v2**: the codebase uses Pydantic v2 (`~=2.10.6`). The MCP extra pins `mcp>=1.9.0,<2.0` to stay within compatible pydantic constraints.

---

## Pull Request Checklist

- [ ] New or changed logic has accompanying tests in `tests/`.
- [ ] `pre-commit run --all-files` passes cleanly.
- [ ] Public-facing changes are documented in `README.md` and/or the relevant `docs/` page.
- [ ] New merge methods are registered in `registry.py` and documented.
- [ ] No new secrets or credentials committed.

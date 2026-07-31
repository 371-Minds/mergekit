# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Model Context Protocol server for mergekit.

Exposes mergekit's model-merging capabilities as MCP tools so AI agents and
MCP-compatible clients can validate configurations, inspect available merge
methods, and run merges programmatically.

Usage
-----
Run directly (stdio transport, default):

    mergekit-mcp

Or via Python:

    python -m mergekit.scripts.mcp_server
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from mcp.server.fastmcp import FastMCP

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS
from mergekit.options import MergeOptions

LOG = logging.getLogger(__name__)

mcp = FastMCP(
    name="mergekit",
    instructions=(
        "MergeKit MCP Server: provides tools for merging pre-trained language models. "
        "Use list_merge_methods to discover available algorithms, "
        "validate_merge_config to check a configuration before running it, "
        "and run_merge_from_config to execute a merge."
    ),
)


# ---------------------------------------------------------------------------
# Tool: list_merge_methods
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "List all merge methods supported by mergekit. "
        "Returns a JSON array of objects, each with 'name', 'pretty_name', "
        "'reference_url', 'parameters', and 'tensor_parameters' fields."
    )
)
def list_merge_methods() -> str:
    """Return all registered merge methods as a JSON string."""
    methods: List[Dict[str, Any]] = []
    for method_name, method in REGISTERED_MERGE_METHODS.items():
        params = [
            {
                "name": p.name,
                "required": p.required,
                "default_value": p.default_value,
            }
            for p in method.parameters()
        ]
        tensor_params = [
            {
                "name": p.name,
                "required": p.required,
                "default_value": p.default_value,
            }
            for p in method.tensor_parameters()
        ]
        methods.append(
            {
                "name": method_name,
                "pretty_name": method.pretty_name() or method_name,
                "reference_url": method.reference_url(),
                "parameters": params,
                "tensor_parameters": tensor_params,
            }
        )
    return json.dumps(methods, indent=2)


# ---------------------------------------------------------------------------
# Tool: get_merge_method_info
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Get detailed information about a specific merge method. "
        "Pass the method's identifier (e.g. 'ties', 'slerp', 'linear'). "
        "Returns a JSON object with 'name', 'pretty_name', 'reference_url', "
        "'parameters', and 'tensor_parameters' fields, or an error message."
    )
)
def get_merge_method_info(method_name: str) -> str:
    """Return information about a single merge method as a JSON string.

    Parameters
    ----------
    method_name:
        The internal name of the merge method (e.g. ``"ties"``).
    """
    method = REGISTERED_MERGE_METHODS.get(method_name)
    if method is None:
        available = sorted(REGISTERED_MERGE_METHODS.keys())
        return json.dumps(
            {
                "error": f"Unknown merge method '{method_name}'.",
                "available_methods": available,
            },
            indent=2,
        )

    params = [
        {
            "name": p.name,
            "required": p.required,
            "default_value": p.default_value,
        }
        for p in method.parameters()
    ]
    tensor_params = [
        {
            "name": p.name,
            "required": p.required,
            "default_value": p.default_value,
        }
        for p in method.tensor_parameters()
    ]
    return json.dumps(
        {
            "name": method_name,
            "pretty_name": method.pretty_name() or method_name,
            "reference_url": method.reference_url(),
            "parameters": params,
            "tensor_parameters": tensor_params,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Tool: validate_merge_config
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Validate a mergekit YAML merge configuration string without running the merge. "
        "Returns a JSON object with 'valid' (bool) and either 'config' (the parsed "
        "configuration as a dict) or 'error' (a validation error message)."
    )
)
def validate_merge_config(config_yaml: str) -> str:
    """Parse and validate a YAML merge configuration.

    Parameters
    ----------
    config_yaml:
        A YAML string containing a mergekit merge configuration.
    """
    try:
        raw = yaml.safe_load(config_yaml)
        cfg = MergeConfiguration.model_validate(raw)
        return json.dumps(
            {
                "valid": True,
                "config": cfg.model_dump(exclude_defaults=True, mode="json"),
            },
            indent=2,
        )
    except yaml.YAMLError as exc:
        return json.dumps({"valid": False, "error": f"YAML parse error: {exc}"}, indent=2)
    except ValidationError as exc:
        return json.dumps(
            {"valid": False, "error": f"Configuration validation error: {exc}"}, indent=2
        )
    except Exception as exc:
        return json.dumps({"valid": False, "error": str(exc)}, indent=2)


# ---------------------------------------------------------------------------
# Tool: run_merge_from_config
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Execute a model merge from a YAML configuration string. "
        "The merged model will be written to 'out_path'. "
        "Accepts optional merge options as a JSON object (keys match MergeOptions fields, "
        "e.g. '{\"cuda\": true, \"copy_tokenizer\": false}'). "
        "Models referenced in the configuration must be accessible from the "
        "environment (local paths or cached Hugging Face models). "
        "Returns a JSON object with 'success' (bool) and either 'out_path' or 'error'."
    )
)
def run_merge_from_config(
    config_yaml: str,
    out_path: str,
    options_json: Optional[str] = None,
) -> str:
    """Run a merge and write the result to *out_path*.

    Parameters
    ----------
    config_yaml:
        A YAML string containing a valid mergekit merge configuration.
    out_path:
        Filesystem path where the merged model should be saved.
    options_json:
        Optional JSON string of :class:`mergekit.options.MergeOptions` fields to
        override the defaults (e.g. ``'{"cuda": true}'``).
    """
    # Parse config
    try:
        raw = yaml.safe_load(config_yaml)
        merge_config = MergeConfiguration.model_validate(raw)
    except yaml.YAMLError as exc:
        return json.dumps({"success": False, "error": f"YAML parse error: {exc}"}, indent=2)
    except ValidationError as exc:
        return json.dumps(
            {"success": False, "error": f"Configuration validation error: {exc}"}, indent=2
        )

    # Parse options
    options_dict: Dict[str, Any] = {}
    if options_json:
        try:
            options_dict = json.loads(options_json)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {"success": False, "error": f"options_json is not valid JSON: {exc}"},
                indent=2,
            )

    try:
        options = MergeOptions(**options_dict)
        options.apply_global_options()
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Invalid merge options: {exc}"}, indent=2
        )

    # Run the merge
    try:
        os.makedirs(out_path, exist_ok=True)
        run_merge(
            merge_config,
            out_path,
            options=options,
            config_source=config_yaml,
        )
        return json.dumps({"success": True, "out_path": out_path}, indent=2)
    except Exception as exc:
        LOG.exception("Merge failed")
        return json.dumps({"success": False, "error": str(exc)}, indent=2)


# ---------------------------------------------------------------------------
# Tool: generate_merge_card
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Generate a model card (README.md content) for a merge configuration "
        "without executing the merge. "
        "Returns a JSON object with 'card' (the markdown string) or 'error'."
    )
)
def generate_merge_card(config_yaml: str, model_name: Optional[str] = None) -> str:
    """Generate a Hugging Face model card for a given merge configuration.

    Parameters
    ----------
    config_yaml:
        A YAML string containing a valid mergekit merge configuration.
    model_name:
        Optional name to use for the model in the card (defaults to ``"merged-model"``).
    """
    from mergekit.card import generate_card

    try:
        raw = yaml.safe_load(config_yaml)
        merge_config = MergeConfiguration.model_validate(raw)
    except yaml.YAMLError as exc:
        return json.dumps({"error": f"YAML parse error: {exc}"}, indent=2)
    except ValidationError as exc:
        return json.dumps(
            {"error": f"Configuration validation error: {exc}"}, indent=2
        )

    try:
        card_md = generate_card(
            config=merge_config,
            config_yaml=config_yaml,
            name=model_name or "merged-model",
        )
        return json.dumps({"card": card_md}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.WARNING)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

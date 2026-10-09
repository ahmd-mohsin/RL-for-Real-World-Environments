"""M3 connectors (the slice M4 needs): raw records -> ``crucible.trace/v1``, redacted inside the tenant.

``base``: the Connector protocol and idempotent ingest; ``trace_store``: traces as content-addressed documents;
``redact``: secrets, emails, internal hosts and actor handles; ``file_drop``: JSONL / CSV event logs.
Domain connectors (GitHub PRs, CI logs, agent sessions) live in packs.
"""

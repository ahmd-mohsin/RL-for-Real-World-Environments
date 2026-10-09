"""M3 connectors: raw records -> ``crucible.trace/v1``, redacted inside the tenant.

``base``: the Connector protocol and idempotent ingest; ``trace_store``: traces as content-addressed documents;
``redact``: credentials (format patterns + the tenant's known values), e-mails, internal hosts, human actors;
``agent_events``: what every agent-session connector agrees on. Generic connectors: ``file_drop`` (JSONL / CSV
event logs), ``sql_audit_log`` (audit tables), ``otel_import`` (OTel GenAI telemetry), ``capture_proxy`` (a
recording proxy in front of live agents, and its log connector); ATIF import lives in ``interop/harbor``.
Domain connectors (GitHub PRs, CI logs, agent sessions, policy docs) live in packs.
"""

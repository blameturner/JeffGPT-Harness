"""Personal-assistant layer for the Home dashboard.

Four tables back this (created in NocoDB):

- ``pa_open_loops``       — things the user said they'd do / are tracking
- ``pa_warm_topics``      — what the PA is "currently on its mind"
- ``pa_user_facts``       — slow-moving identity: role, preferences, routines
- ``pa_assistant_moves``  — log of proactive/inline surfaces, for rotation

All writes are best-effort: when a table is missing, helpers log a warning
and return empty / no-op. Feature-gated behind ``pa.enabled``.
"""

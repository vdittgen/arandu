"""Versioned config defaults consumed by the egress firewall.

The YAML files in this package ship as the in-repo source of truth.
Both ``pricing.yaml`` and ``spend_caps.yaml`` are overrideable per
user at ``~/.arandu/config/pricing.override.yaml`` /
``spend_caps.override.yaml`` so the runtime can be tuned without
forking the repo.

sensitivity_tier: 1
"""

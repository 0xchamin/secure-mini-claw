package secureclaw.authz

import rego.v1

default allow := false
default reason := "denied by default"
default timeout_seconds := 30

# -----------------------------------------------
# TBAC: per-skill tool whitelist
# -----------------------------------------------

skill_allowed_tools := {
    "mcp-watcher":        {"github_list_issues", "alert_send"},
    "voyageintel-spike":  {"flights_near", "vessels_near", "alert_telegram"},
    "core":               {"*"},
}

# Allow if the tool is in the skill's whitelist or skill has wildcard
allow if {
    tools := skill_allowed_tools[input.skill]
    tools["*"]
}

allow if {
    tools := skill_allowed_tools[input.skill]
    input.tool in tools
}

# -----------------------------------------------
# ABAC: time-of-day restriction example
# -----------------------------------------------

# Deny elevated tools outside business hours (8-20 UTC)
elevated_tools := {"alert_telegram", "alert_send"}

allow := false if {
    input.tool in elevated_tools
    hour := time.clock(input.timestamp * 1000000000)[0]
    hour < 8
}

allow := false if {
    input.tool in elevated_tools
    hour := time.clock(input.timestamp * 1000000000)[0]
    hour > 20
}

# -----------------------------------------------
# Reason + timeout overrides
# -----------------------------------------------

reason := "tool not in skill whitelist" if { not allow }
reason := "allowed by TBAC" if { allow }

timeout_seconds := 10 if { input.tool in elevated_tools }
timeout_seconds := 60 if { input.data_sensitivity == "high" }

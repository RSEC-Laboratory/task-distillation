predicate Critical(event):
    event.risk == TRUE or event.action == delete

predicate Trusted(event):
    event.role == auditor or (event.context == internal and event.risk == FALSE)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternal:
    when event.context == sandbox and event.risk == UNKNOWN
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedChange:
    when Trusted(event)
    then forbid label = review

rule InspectSandbox:
    when event.action == inspect and event.context == sandbox
    then prefer label = allow priority 70

rule AnalystPartner:
    when event.role == analyst and event.context == partner
    then prefer label = review priority 40

rule DefaultDeny:
    when event.role == auditor or event.role != auditor
    then prefer label = deny priority 1
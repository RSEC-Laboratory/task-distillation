predicate Critical(event):
    event.risk == TRUE or event.action == delete and event.context == sandbox

predicate Trusted(event):
    event.role == admin or event.role == analyst

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == sandbox
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedInspect:
    when Trusted(event) and event.action == inspect
    then prefer label = allow priority 80

rule TrustedDeny:
    when Trusted(event)
    then forbid label = deny

rule ApproveSandbox:
    when event.action == approve and event.context == sandbox
    then prefer label = review priority 70

rule AnalystInternal:
    when event.role == analyst or event.context == internal
    then prefer label = deny priority 40

rule DefaultAllow:
    when event.role == admin or event.role != admin
    then prefer label = allow priority 1

predicate Critical(event):
    (event.risk == TRUE and event.role == guest) or event.action == approve

predicate Trusted(event):
    event.context == partner and event.role!= guest

predicate UnknownRisk(event):
    event.risk == UNKNOWN

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when UnknownRisk(event) and event.context == internal
    then abstain

rule UnknownRiskDefault:
    when UnknownRisk(event)
    then prefer label = allow priority 90

rule TrustedDelete:
    when Trusted(event) and event.action == delete
    then prefer label = review priority 80

rule ChangeInInternal:
    when event.action == change and event.context == internal
    then prefer label = deny priority 70

rule AuditorInSandbox:
    when event.role == auditor and event.context == sandbox
    then prefer label = allow priority 40

rule TrustedForbidAllow:
    when Trusted(event)
    then forbid label = allow

rule OtherRoles:
    when event.role in {operator, auditor}
    then prefer label = review priority 1
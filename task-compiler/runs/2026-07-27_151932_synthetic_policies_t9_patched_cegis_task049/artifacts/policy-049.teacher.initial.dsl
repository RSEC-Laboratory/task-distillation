predicate Critical(event):
    event.risk == TRUE and event.context == internal or event.action == export

predicate Trusted(event):
    event.role == auditor or event.context == partner and event.risk == FALSE

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate UnknownExternal(event):
    event.context == internal and UnknownRisk(event)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when UnknownRisk(event) and not UnknownExternal(event)
    then prefer label = review priority 90

rule TrustedInspect:
    when Trusted(event) and event.action == inspect
    then prefer label = deny priority 80, forbid label = review

rule DeleteInternal:
    when event.action == delete and event.context == internal
    then prefer label = allow priority 70

rule AnalystExternal:
    when event.role == analyst and event.context == external
    then prefer label = review priority 40

rule OtherRoles:
    when event.role!= auditor and event.role!= analyst and event.role!= operator
    then prefer label = deny priority 1
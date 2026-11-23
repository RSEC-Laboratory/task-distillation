predicate Critical(event):
    (event.risk == TRUE and event.role == analyst) or event.action == approve

predicate Trusted(event):
    event.context == sandbox and event.role!= analyst

predicate UnknownExternal(event):
    event.risk == UNKNOWN and event.context == internal

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedChange:
    when Trusted(event)
    then prefer label = review priority 80, forbid label = deny

rule InspectInternal:
    when event.action == inspect and event.context == internal
    then prefer label = allow priority 70

rule OperatorPartner:
    when event.role == operator and event.context == partner
    then prefer label = deny priority 40

rule OtherRoles:
    when event.role in {auditor, analyst}
    then prefer label = review priority 1
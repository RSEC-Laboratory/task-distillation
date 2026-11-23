predicate Critical(event):
    (event.risk == TRUE) or (event.action == export)

predicate Trusted(event):
    (event.role == auditor) and (event.context != internal)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == internal)

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedDelete:
    when (Trusted(event)) and (event.action == delete)
    then prefer label = allow priority 80

rule InspectInternal:
    when (event.action == inspect) and (event.context == internal)
    then prefer label = deny priority 70

rule AnalystOrExternal:
    when (event.role == analyst) or (event.context == external)
    then prefer label = review priority 40

rule TrustedForbidReview:
    when Trusted(event)
    then forbid label = review

rule OtherRoles:
    when event.role in {auditor, analyst, operator}
    then prefer label = allow priority 1

predicate Critical(event):
    ((event.risk == TRUE) and (event.role == admin)) or (event.action == export)

predicate Trusted(event):
    (event.context == internal) and (event.role != admin)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == sandbox)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedApprove:
    when (Trusted(event)) and (event.action == approve)
    then prefer label = review priority 80

rule InspectSandbox:
    when (event.action == inspect) and (event.context == sandbox)
    then prefer label = allow priority 70

rule OperatorPartner:
    when (event.role == operator) and (event.context == partner)
    then prefer label = deny priority 40

rule TrustedForbidDeny:
    when Trusted(event)
    then forbid label = deny

rule OtherRoles:
    when event.role in {auditor, operator}
    then prefer label = review priority 1

rule AdminReview:
    when event.role == admin
    then prefer label = review priority 1

predicate Critical(event):
    ((event.risk == TRUE) and (event.role == operator)) or (event.action == export)

predicate Trusted(event):
    (event.context == sandbox) and (event.role != operator)

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate UnknownExternal(event):
    (event.context == external) and (UnknownRisk(event))

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when (UnknownRisk(event)) and (not (UnknownExternal(event)))
    then prefer label = allow priority 90

rule TrustedDelete:
    when (Trusted(event)) and (event.action == delete)
    then prefer label = review priority 80

rule InspectExternal:
    when (event.action == inspect) and (event.context == external)
    then prefer label = deny priority 70

rule GuestPartner:
    when (event.role == guest) and (event.context == partner)
    then prefer label = allow priority 40

rule TrustedForbidAllow:
    when Trusted(event)
    then forbid label = allow

rule OtherRoles:
    when event.role in {analyst, operator}
    then prefer label = review priority 1

rule GuestOther:
    when event.role == guest
    then prefer label = review priority 1

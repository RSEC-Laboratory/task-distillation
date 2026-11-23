predicate Critical(event):
    (event.risk == TRUE) or ((event.action == export) and (event.context == internal))

predicate Trusted(event):
    (event.role == guest) or (event.role == admin)

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when (event.risk == UNKNOWN) and (event.context == internal)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedChange:
    when ((Trusted(event)) and (event.action != export)) and (event.risk != FALSE)
    then prefer label = deny priority 80

rule InspectInternal:
    when (event.action == inspect) and (event.context == internal)
    then prefer label = review priority 70

rule TrustedForbidAllow:
    when (Trusted(event)) and (event.action != export)
    then forbid label = allow

rule AdminOrPartner:
    when (event.role == admin) or (event.context == partner)
    then prefer label = allow priority 40

rule DefaultDeny:
    when (event.role == guest) or (event.role != guest)
    then prefer label = deny priority 1

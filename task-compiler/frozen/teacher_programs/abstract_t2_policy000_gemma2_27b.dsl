predicate Critical(event):
    event.risk == TRUE or event.action == change

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == internal
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedExport:
    when event.role == admin and event.context != internal
    then prefer label = allow priority 80

rule TrustedForbidDeny:
    when event.role == admin
    then forbid label = deny

rule InspectInternal:
    when event.action == inspect and event.context == internal
    then prefer label = review priority 70

rule OperatorPartner:
    when event.role == operator or event.context == partner
    then prefer label = deny priority 40

rule DefaultAllow:
    when event.role == admin or event.role != admin
    then prefer label = allow priority 1

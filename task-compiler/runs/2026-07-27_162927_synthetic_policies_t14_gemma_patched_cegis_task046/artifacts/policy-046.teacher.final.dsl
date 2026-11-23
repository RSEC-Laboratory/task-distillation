predicate Critical(event):
    (event.risk == TRUE) or ((event.action == approve) and (event.context == external))

predicate Trusted(event):
    (event.role == guest) or (event.role == auditor)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternal:
    when (event.risk == UNKNOWN) and (event.context == external)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedInspect:
    when (Trusted(event)) and (event.action == inspect)
    then prefer label = review priority 80

rule TrustedForbidDeny:
    when (Trusted(event)) and (event.action != approve)
    then forbid label = deny

rule ActionChangeExternal:
    when (event.action == change) and (event.context == external)
    then prefer label = allow priority 70

rule RoleAuditorPartner:
    when (event.role == auditor) or (event.context == partner)
    then prefer label = deny priority 40

rule Default:
    when (event.role == guest) or (event.role != guest)
    then prefer label = review priority 1

predicate Critical(event):
    (event.risk == TRUE) or ((event.action == delete) and (event.context == external))

predicate Trusted(event):
    ((event.role == guest) or (event.role == operator)) and (event.action != delete)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == external)

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when (event.risk == UNKNOWN) and (not (UnknownExternal(event)))
    then prefer label = allow priority 90

rule TrustedChange:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = allow

rule ExportExternal:
    when (event.action == export) and (event.context == external)
    then prefer label = review priority 70

rule OperatorOrPartner:
    when (event.role == operator) or (event.context == partner)
    then prefer label = allow priority 40

rule OtherRoles:
    when ((event.role != guest) and (event.role != operator)) and (event.role != analyst)
    then prefer label = deny priority 1

rule AnalystDeny:
    when event.role == analyst
    then prefer label = deny priority 1

rule ReviewOverride:
    when ((event.action == export) and (event.context == external)) and ((event.role == operator) or (event.role == guest))
    then prefer label = review priority 100

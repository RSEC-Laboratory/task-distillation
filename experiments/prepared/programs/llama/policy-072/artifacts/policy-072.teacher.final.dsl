predicate Critical(event):
    (event.risk == TRUE) or (event.action == inspect)

predicate Trusted(event):
    (event.role == operator) and (event.context != internal)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == internal)

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when ((event.risk == UNKNOWN) and (not (UnknownExternal(event)))) and (event.role != operator)
    then prefer label = allow priority 90

rule TrustedApprove:
    when (Trusted(event)) and (event.action == approve)
    then prefer label = review priority 80, forbid label = allow

rule DeleteInternal:
    when (event.action == delete) and (event.context == internal)
    then prefer label = deny priority 70

rule AdminOrExternal:
    when ((event.role == admin) or (event.context == external)) and (event.role != operator)
    then prefer label = allow priority 40

rule OtherRoles:
    when event.role in {operator, analyst}
    then prefer label = review priority 1

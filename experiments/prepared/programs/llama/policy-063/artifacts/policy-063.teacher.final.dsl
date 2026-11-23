predicate Critical(event):
    ((event.risk == TRUE) and (event.role == admin)) or (event.action == inspect)

predicate Trusted(event):
    (event.context == partner) and (event.role != admin)

predicate UnknownExternal(event):
    (event.context == internal) and (event.risk == UNKNOWN)

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when (event.risk == UNKNOWN) and (not (UnknownExternal(event)))
    then prefer label = deny priority 90

rule TrustedExport:
    when Trusted(event)
    then prefer label = allow priority 80, forbid label = deny

rule InternalChange:
    when (event.action == change) and (event.context == internal)
    then prefer label = review priority 70

rule GuestExternal:
    when (event.role == guest) and (event.context == external)
    then prefer label = deny priority 40

rule OtherRoles:
    when event.role in {auditor, admin}
    then prefer label = allow priority 1

rule GuestInternal:
    when (event.role == guest) and (event.context == internal)
    then prefer label = allow priority 1

predicate Critical(event):
    (event.risk == TRUE) or (event.action == change)

predicate Trusted(event):
    (event.role == operator) and (event.context != internal)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == internal)

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when (event.risk == UNKNOWN) and (not (UnknownExternal(event)))
    then prefer label = deny priority 90

rule TrustedApprove:
    when (Trusted(event)) and (event.action == approve)
    then prefer label = allow priority 80, forbid label = deny

rule ExportInternal:
    when (event.action == export) and (event.context == internal)
    then prefer label = review priority 70

rule AdminOrPartner:
    when (event.role == admin) or (event.context == partner)
    then prefer label = deny priority 40

rule TrustedExport:
    when (Trusted(event)) and (event.action == export)
    then prefer label = allow priority 80

rule OtherRoles:
    when (event.role != admin) and (event.role != guest)
    then prefer label = allow priority 1

predicate Critical(event):
    (event.risk == TRUE) or ((event.action == export) and (event.context == internal))

predicate Trusted(event):
    ((event.role == guest) or (event.role == admin)) and (event.action != export)

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
    then prefer label = allow priority 90

rule TrustedChange:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = allow

rule InspectInternal:
    when (event.action == inspect) and (event.context == internal)
    then prefer label = review priority 70

rule AdminOrPartner:
    when (event.role == admin) or (event.context == partner)
    then prefer label = allow priority 40

rule OtherRoles:
    when event.role in {operator, guest, admin}
    then prefer label = deny priority 1

rule GuestInspectInternal:
    when ((event.role == guest) and (event.action == inspect)) and (event.context == internal)
    then prefer label = review priority 100

rule SourceCounterexamplePatch1:
    when (((event.role == admin) and (event.action == inspect)) and (event.context == internal)) and (event.risk == FALSE)
    then require label = review

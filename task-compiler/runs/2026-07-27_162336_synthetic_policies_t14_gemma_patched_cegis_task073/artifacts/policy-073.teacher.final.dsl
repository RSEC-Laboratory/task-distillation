predicate Critical(event):
    (event.risk == TRUE) or (event.action == approve)

predicate Trusted(event):
    (event.role == auditor) or ((event.context == partner) and (event.risk == FALSE))

rule CriticalCase:
    when (((Critical(event)) and (event.action != delete)) and (not (((event.action == export) and (event.context == partner)) and (event.risk == TRUE)))) and (not (((event.action == export) and (event.context == external)) and (event.risk == TRUE)))
    then require label = review

rule UnknownExternal:
    when ((event.risk == UNKNOWN) and (event.context == external)) and (event.risk != UNKNOWN)
    then abstain

rule UnknownRisk:
    when (event.risk == UNKNOWN) and (not ((event.context == sandbox) and (event.risk == UNKNOWN)))
    then prefer label = deny priority 90

rule TrustedCase:
    when Trusted(event)
    then forbid label = deny

rule DeleteInSandbox:
    when ((event.action == delete) and (event.context == sandbox)) and (not ((event.context == sandbox) and (event.risk == UNKNOWN)))
    then prefer label = review priority 70

rule GuestExternal:
    when (event.role == guest) and (event.context == external)
    then prefer label = deny priority 40

rule Default:
    when ((event.role == auditor) or (event.role != auditor)) and (not ((event.context == sandbox) and (event.risk == UNKNOWN)))
    then prefer label = allow priority 1

rule SourceCounterexamplePatch1:
    when (event.context == sandbox) and (event.risk == UNKNOWN)
    then abstain

rule SourceCounterexamplePatch2:
    when ((event.action == export) and (event.context == partner)) and (event.risk == TRUE)
    then require label = allow

rule SourceCounterexamplePatch3:
    when (((event.action == export) and (event.context == external)) and (event.risk == TRUE)) and (not (((event.role == auditor) and (event.action == export)) and (event.context == external)))
    then require label = deny

rule SourceCounterexamplePatch4:
    when ((event.role == auditor) and (event.action == export)) and (event.context == external)
    then require label = allow

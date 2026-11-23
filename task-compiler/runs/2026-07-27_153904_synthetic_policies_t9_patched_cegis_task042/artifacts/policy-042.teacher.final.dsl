predicate Critical(event):
    (event.risk == TRUE) or ((event.action == change) and (event.context == internal))

predicate Trusted(event):
    ((event.role == analyst) or (event.role == auditor)) and (event.action != change)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == internal)

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedApprove:
    when (Trusted(event)) and (event.action == approve)
    then prefer label = review priority 80, forbid label = allow

rule DeleteInternal:
    when (event.action == delete) and (event.context == internal)
    then prefer label = deny priority 70

rule AuditorOrSandbox:
    when (event.role == auditor) or (event.context == sandbox)
    then prefer label = allow priority 40

rule OtherRoles:
    when event.role != guest
    then prefer label = review priority 1

rule GuestRole:
    when event.role == guest
    then prefer label = review priority 1

rule SourceCounterexamplePatch1:
    when (((event.role == auditor) and (event.action == delete)) and (event.context == sandbox)) and (event.risk == FALSE)
    then require label = review

rule SourceCounterexamplePatch2:
    when (((event.role == auditor) and (event.action == delete)) and (event.context == external)) and (event.risk == UNKNOWN)
    then require label = review

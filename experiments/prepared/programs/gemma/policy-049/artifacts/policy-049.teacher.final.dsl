predicate Critical(event):
    (event.risk == TRUE) and (event.context == internal)

predicate Trusted(event):
    (event.role == auditor) or ((event.context == partner) and (event.risk == FALSE))

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternal:
    when (event.risk == UNKNOWN) and (event.context == internal)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedInspect:
    when (Trusted(event)) and (event.action == inspect)
    then forbid label = review

rule DeleteInternal:
    when (event.action == delete) and (event.context == internal)
    then prefer label = allow priority 70

rule AnalystExternal:
    when (event.role == analyst) and (event.context == external)
    then prefer label = review priority 40

rule TrustedForbidReview:
    when Trusted(event)
    then forbid label = review

rule DefaultDeny:
    when (event.role == auditor) or (event.role != auditor)
    then prefer label = deny priority 1

rule SourceCounterexamplePatch1:
    when event.action == export
    then require label = allow

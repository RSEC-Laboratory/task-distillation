predicate Critical(event):
    (event.risk == TRUE) or (event.action == delete)

predicate Trusted(event):
    (event.role == auditor) or ((event.context == internal) and (event.risk == FALSE))

rule CriticalCase:
    when (((Critical(event)) and (event.action != inspect)) and (not (((event.action == change) and (event.context == internal)) and (event.risk == TRUE)))) and (not (((event.action == change) and (event.context == partner)) and (event.risk == TRUE)))
    then require label = allow

rule UnknownExternal:
    when (event.context == sandbox) and (event.risk == UNKNOWN)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedChange:
    when Trusted(event)
    then forbid label = review

rule InspectSandbox:
    when (event.action == inspect) and (event.context == sandbox)
    then prefer label = allow priority 70

rule AnalystPartner:
    when (event.role == analyst) and (event.context == partner)
    then prefer label = review priority 40

rule DefaultDeny:
    when (event.role == auditor) or (event.role != auditor)
    then prefer label = deny priority 1

rule SourceCounterexamplePatch1:
    when ((event.action == change) and (event.context == internal)) and (event.risk == TRUE)
    then require label = deny

rule SourceCounterexamplePatch2:
    when (((event.action == change) and (event.context == partner)) and (event.risk == TRUE)) and (not (((event.role == analyst) and (event.action == change)) and (event.context == partner)))
    then require label = deny

rule SourceCounterexamplePatch3:
    when ((event.role == analyst) and (event.action == change)) and (event.context == partner)
    then require label = review

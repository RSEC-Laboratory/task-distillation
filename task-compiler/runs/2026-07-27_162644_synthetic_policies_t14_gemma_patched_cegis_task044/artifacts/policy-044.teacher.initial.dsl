predicate Critical(event):
    event.risk == TRUE or event.action == export

predicate Trusted(event):
    event.role == auditor and event.context != internal

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == internal
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedDelete:
    when Trusted(event) and event.action == delete
    then prefer label = allow priority 80
rule TrustedForbidReview:
    when Trusted(event)
    then forbid label = review

rule InspectInternal:
    when event.action == inspect and event.context == internal
    then prefer label = deny priority 70

rule AnalystExternal:
    when event.role == analyst or event.context == external
    then prefer label = review priority 40

rule DefaultAllow:
    when event.role == auditor or event.role != auditor
    then prefer label = allow priority 1
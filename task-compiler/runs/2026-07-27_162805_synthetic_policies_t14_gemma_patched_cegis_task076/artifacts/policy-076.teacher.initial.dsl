predicate Critical(event):
    event.risk == TRUE or event.action == approve

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternal:
    when event.context == external and event.risk == UNKNOWN
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedChange:
    when event.role == admin and event.context != external
    then prefer label = deny priority 80

rule ExportExternal:
    when event.context == external and event.action == export
    then prefer label = allow priority 70

rule AnalystOrPartner:
    when event.role == analyst or event.context == partner
    then prefer label = review priority 40

rule Default:
    when event.role == admin or event.role != admin
    then prefer label = deny priority 1

rule ForbidReview:
    when Critical(event)
    then forbid label = review
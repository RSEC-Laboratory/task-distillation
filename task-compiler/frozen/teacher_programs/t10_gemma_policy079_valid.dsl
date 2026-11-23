predicate Critical(event):
    event.risk == TRUE and event.role == analyst or event.action == delete

predicate Trusted(event):
    event.context == sandbox and event.role != analyst

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.context == partner and event.risk == UNKNOWN
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedChange:
    when Trusted(event)
    then prefer label = deny priority 80

rule InspectPartner:
    when event.action == inspect and event.context == partner
    then prefer label = review priority 70

rule AuditorExternal:
    when event.role == auditor and event.context == external
    then prefer label = allow priority 40

rule DefaultDeny:
    when event.role == operator or event.role != operator
    then prefer label = deny priority 1

rule ForbidAllow:
    when Trusted(event)
    then forbid label = allow

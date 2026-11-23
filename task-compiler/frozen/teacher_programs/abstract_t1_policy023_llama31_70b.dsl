predicate Critical(event):
    (event.risk == TRUE and event.role == guest) or event.action == inspect

predicate Trusted(event):
    event.context == partner and event.role != guest

predicate UnknownExternal(event):
    event.context == external and event.risk == UNKNOWN

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedApprove:
    when Trusted(event) and event.action == approve
    then prefer label = deny priority 80

rule DeleteSandbox:
    when event.action == delete and event.context == sandbox
    then prefer label = review priority 70

rule AuditorExternal:
    when event.role == auditor and event.context == external
    then prefer label = allow priority 40

rule TrustedForbidAllow:
    when Trusted(event)
    then forbid label = allow

rule OtherRoles:
    when event.role in {analyst, guest}
    then prefer label = deny priority 1

predicate Critical(event):
    event.risk == TRUE and event.role == guest or event.action == inspect

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == sandbox
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedApprove:
    when event.context == partner and event.role != guest
    then forbid label = allow

rule TrustedApproveDeny:
    when event.context == partner and event.role != guest and event.action == approve
    then prefer label = deny priority 80

rule DeleteSandbox:
    when event.action == delete and event.context == sandbox
    then prefer label = review priority 70

rule AuditorExternal:
    when event.role == auditor and event.context == external
    then prefer label = allow priority 40

rule DefaultDeny:
    when event.role == analyst or event.role != analyst
    then prefer label = deny priority 1

predicate Critical(event):
    event.risk == TRUE and event.role == guest or event.action == approve

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == internal
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedDelete:
    when event.context == partner and event.role != guest and event.action == delete
    then prefer label = review priority 80
    
rule TrustedForbidAllow:
    when event.context == partner and event.role != guest
    then forbid label = allow

rule ActionChangeInternal:
    when event.action == change and event.context == internal
    then prefer label = deny priority 70

rule AuditorSandbox:
    when event.role == auditor and event.context == sandbox
    then prefer label = allow priority 40

rule Default:
    when event.role == operator or event.role != operator
    then prefer label = review priority 1
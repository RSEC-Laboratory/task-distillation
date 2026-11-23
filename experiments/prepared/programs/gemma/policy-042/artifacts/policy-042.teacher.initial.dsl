predicate Critical(event):
    event.risk == TRUE or (event.action == change and event.context == internal)

predicate Trusted(event):
    event.role == analyst or event.role == auditor

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == external
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedApprove:
    when Trusted(event) and event.action != change
    then prefer label = review priority 80
    
rule DeleteInternal:
    when event.action == delete and event.context == internal
    then prefer label = deny priority 70

rule AuditorSandbox:
    when event.role == auditor or event.context == sandbox
    then prefer label = allow priority 40

rule TrustedForbidAllow:
    when Trusted(event)
    then forbid label = allow

rule Default:
    when event.role == analyst or event.role != analyst
    then prefer label = review priority 1
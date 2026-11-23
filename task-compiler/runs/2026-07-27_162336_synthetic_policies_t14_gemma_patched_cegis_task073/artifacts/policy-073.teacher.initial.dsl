predicate Critical(event):
    event.risk == TRUE or event.action == approve

predicate Trusted(event):
    event.role == auditor or (event.context == partner and event.risk == FALSE)

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == external
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedCase:
    when Trusted(event)
    then forbid label = deny

rule DeleteInSandbox:
    when event.action == delete and event.context == sandbox
    then prefer label = review priority 70

rule GuestExternal:
    when event.role == guest and event.context == external
    then prefer label = deny priority 40

rule Default:
    when event.role == auditor or event.role != auditor
    then prefer label = allow priority 1
predicate Critical(event):
    event.risk == TRUE or event.action == export

predicate Trusted(event):
    event.role == admin and event.context != internal

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == internal
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedChange:
    when Trusted(event)
    then forbid label = allow

rule ApproveInternal:
    when event.action == approve and event.context == internal
    then prefer label = review priority 70

rule OperatorOrSandbox:
    when event.role == operator or event.context == sandbox
    then prefer label = allow priority 40

rule DefaultDeny:
    when event.role == admin or event.role != admin
    then prefer label = deny priority 1
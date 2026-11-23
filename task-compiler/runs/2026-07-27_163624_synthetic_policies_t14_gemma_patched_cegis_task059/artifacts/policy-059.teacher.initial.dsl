predicate Critical(event):
    event.risk == TRUE and event.role == admin or event.action == export

predicate Trusted(event):
    event.context == internal and event.role != admin

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule TrustedApprove:
    when Trusted(event)
    then prefer label = review priority 80

rule UnknownExternal:
    when event.context == sandbox and event.risk == UNKNOWN
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule InspectSandbox:
    when event.action == inspect and event.context == sandbox
    then prefer label = allow priority 70

rule OperatorPartner:
    when event.role == operator and event.context == partner
    then prefer label = deny priority 40

rule Default:
    when event.role == auditor or event.role != auditor
    then prefer label = review priority 1
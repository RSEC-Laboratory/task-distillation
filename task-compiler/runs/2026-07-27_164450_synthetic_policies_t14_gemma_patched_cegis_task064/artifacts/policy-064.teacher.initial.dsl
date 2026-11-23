predicate Critical(event):
    event.risk == TRUE or event.action == approve

predicate Trusted(event):
    event.role == guest and event.context != partner

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == partner
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedChange:
    when Trusted(event)
    then forbid label = review

rule ActionExport:
    when event.action == export and event.context == partner
    then prefer label = deny priority 70

rule AnalystOrSandbox:
    when event.role == analyst or event.context == sandbox
    then prefer label = review priority 40

rule Default:
    when event.role == guest or event.role != guest
    then prefer label = allow priority 1
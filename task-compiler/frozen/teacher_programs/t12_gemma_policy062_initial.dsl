predicate Critical(event):
    event.risk == TRUE or event.action == change and event.context == sandbox

predicate Trusted(event):
    event.role == guest or event.role == admin

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == sandbox
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedExport:
    when Trusted(event) and event.action != change
    then prefer label = allow priority 80

rule InspectSandbox:
    when event.action == inspect and event.context == sandbox
    then prefer label = deny priority 70

rule AdminExternalReview:
    when event.role == admin or event.context == external
    then prefer label = review priority 40

rule DefaultAllow:
    when event.role == guest or event.role != guest
    then prefer label = allow priority 1

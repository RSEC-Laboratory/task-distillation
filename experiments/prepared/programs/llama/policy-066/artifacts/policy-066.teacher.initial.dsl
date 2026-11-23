predicate Critical(event):
    event.risk == TRUE or (event.action == approve and event.context == external)

predicate Trusted(event):
    (event.role == operator or event.role == analyst) and event.action!= approve

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == external
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedExport:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = allow

rule ExternalChange:
    when event.action == change and event.context == external
    then prefer label = review priority 70

rule AnalystOrPartner:
    when event.role == analyst or event.context == partner
    then prefer label = allow priority 40

rule OtherRoles:
    when event.role == admin
    then prefer label = deny priority 1
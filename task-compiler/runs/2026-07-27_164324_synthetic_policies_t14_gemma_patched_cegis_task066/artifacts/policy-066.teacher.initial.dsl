predicate Critical(event):
    event.risk == TRUE or event.action == approve and event.context == external

predicate Trusted(event):
    event.role == operator or event.role == analyst

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == external
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = allow priority 90

rule TrustedDeny:
    when Trusted(event) and not event.action == approve
    then prefer label = deny priority 80

rule ActionChangeExternal:
    when event.action == change and event.context == external
    then prefer label = review priority 70

rule AnalystOrPartner:
    when event.role == analyst or event.context == partner
    then prefer label = allow priority 40

rule DefaultDeny:
    when event.role == operator or event.role != operator
    then prefer label = deny priority 1
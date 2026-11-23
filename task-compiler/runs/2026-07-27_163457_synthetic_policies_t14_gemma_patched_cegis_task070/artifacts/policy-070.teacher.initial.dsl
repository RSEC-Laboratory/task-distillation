predicate Critical(event):
    event.risk == TRUE or event.action == delete and event.context == external

predicate Trusted(event):
    event.role == guest or event.role == operator

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == external
    then abstain
rule TrustedChange:
    when Trusted(event) and event.action != delete
    then forbid label = allow

rule ExportReview:
    when event.action == export and event.context == external
    then prefer label = review priority 70

rule TrustedDeny:
    when Trusted(event)
    then prefer label = deny priority 80

rule PartnerAllow:
    when event.context == partner or event.role == operator
    then prefer label = allow priority 40

rule DefaultDeny:
    when event.role == guest or event.role != guest
    then prefer label = deny priority 1
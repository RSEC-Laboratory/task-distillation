predicate Critical(event):
    event.risk == TRUE or (event.action == change and event.context == sandbox)

predicate Trusted(event):
    (event.role == guest or event.role == admin) and event.action!= change

predicate UnknownExternal(event):
    event.risk == UNKNOWN and event.context == external

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedExport:
    when Trusted(event) and event.action == export
    then prefer label = allow priority 80

rule InspectSandbox:
    when event.action == inspect and event.context == sandbox
    then prefer label = deny priority 70

rule AdminOrExternal:
    when event.role == admin or event.context == external
    then prefer label = review priority 40

rule TrustedForbidReview:
    when Trusted(event)
    then forbid label = review

rule OtherRoles:
    when event.role in {guest, admin, operator}
    then prefer label = allow priority 1
predicate Critical(event):
    event.risk == TRUE and event.role == operator or event.action == export

predicate Trusted(event):
    event.context == sandbox and event.role != operator

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.context == external and event.risk == UNKNOWN
    then abstain

rule TrustedDelete:
    when Trusted(event) and event.action == delete
    then prefer label = review priority 80

rule InspectExternal:
    when event.context == external and event.action == inspect
    then prefer label = deny priority 70

rule GuestPartner:
    when event.role == guest and event.context == partner
    then prefer label = allow priority 40

rule Default:
    when event.role == analyst or event.role != analyst
    then prefer label = review priority 1
predicate Critical(event):
    event.risk == TRUE or event.action == change

predicate Trusted(event):
    event.context == sandbox and event.role != analyst

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == internal
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedDelete:
    when Trusted(event) and event.action == delete
    then prefer label = allow priority 80

rule ActionExportInternal:
    when event.action == export and event.context == internal
    then prefer label = deny priority 70

rule AdminPartner:
    when event.role == admin and event.context == partner
    then prefer label = review priority 40

rule Default:
    when event.role == operator or event.role != operator
    then prefer label = allow priority 1
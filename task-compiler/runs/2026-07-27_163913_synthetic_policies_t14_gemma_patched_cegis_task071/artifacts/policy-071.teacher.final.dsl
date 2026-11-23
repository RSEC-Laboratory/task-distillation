predicate Critical(event):
    (event.risk == TRUE) or (event.action == change)

predicate Trusted(event):
    (event.context == sandbox) and (event.role != analyst)

rule CriticalCase:
    when ((((Critical(event)) and (not (((event.role == operator) and (event.action == delete)) and (event.risk == TRUE)))) and (not (((event.role == admin) and (event.action == delete)) and (event.context == partner)))) and (not (((event.role == admin) and (event.action == export)) and (event.context == partner)))) and (not (((event.role == admin) and (event.action == export)) and (event.context == sandbox)))
    then require label = deny

rule UnknownExternal:
    when (event.risk == UNKNOWN) and (event.context == internal)
    then abstain

rule UnknownRisk:
    when (event.risk == UNKNOWN) and (event.role != admin)
    then prefer label = review priority 90

rule TrustedDelete:
    when (Trusted(event)) and (event.action == delete)
    then prefer label = allow priority 80

rule ActionExportInternal:
    when (event.action == export) and (event.context == internal)
    then prefer label = deny priority 70

rule AdminPartner:
    when (event.role == admin) and (event.context == partner)
    then prefer label = review priority 40

rule Default:
    when (event.role == operator) or (event.role != operator)
    then prefer label = allow priority 1

rule SourceCounterexamplePatch1:
    when ((event.role == operator) and (event.action == delete)) and (event.risk == TRUE)
    then require label = allow

rule SourceCounterexamplePatch2:
    when ((event.role == admin) and (event.action == delete)) and (event.context == partner)
    then require label = review

rule SourceCounterexamplePatch3:
    when ((event.role == admin) and (event.action == export)) and (event.context == partner)
    then require label = review

rule SourceCounterexamplePatch4:
    when ((event.role == admin) and (event.action == export)) and (event.context == sandbox)
    then require label = allow

predicate Critical(event):
    ((event.risk == TRUE) and (event.context == sandbox)) or (event.action == delete)

predicate Trusted(event):
    (event.role == auditor) or ((event.context == internal) and (event.risk == FALSE))

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == sandbox)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedChange:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = review

rule InspectSandbox:
    when (event.action == inspect) and (event.context == sandbox)
    then prefer label = allow priority 70

rule AnalystPartner:
    when (event.role == analyst) and (event.context == partner)
    then prefer label = review priority 40

rule OtherAnalyst:
    when event.role == analyst
    then prefer label = deny priority 1

rule GuestRole:
    when event.role == guest
    then prefer label = deny priority 1

rule OtherRoles:
    when ((event.role != auditor) and (event.role != analyst)) and (event.role != guest)
    then prefer label = deny priority 1

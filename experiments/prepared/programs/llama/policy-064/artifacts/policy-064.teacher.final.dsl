predicate Critical(event):
    (event.risk == TRUE) or (event.action == approve)

predicate Trusted(event):
    (event.role == guest) and (event.context != partner)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == partner)

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when (event.risk == UNKNOWN) and (not (UnknownExternal(event)))
    then prefer label = review priority 90

rule TrustedChange:
    when Trusted(event)
    then prefer label = allow priority 80, forbid label = review

rule ExportInPartner:
    when (event.action == export) and (event.context == partner)
    then prefer label = deny priority 70

rule AnalystOrSandbox:
    when (event.role == analyst) or (event.context == sandbox)
    then prefer label = review priority 40

rule OtherDeclaredRoles:
    when (event.role == admin) or (event.role == guest)
    then prefer label = allow priority 1

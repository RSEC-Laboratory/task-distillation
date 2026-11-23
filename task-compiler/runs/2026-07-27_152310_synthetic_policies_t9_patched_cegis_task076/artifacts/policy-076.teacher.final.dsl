predicate Critical(event):
    (event.risk == TRUE) or (event.action == approve)

predicate Trusted(event):
    (event.role == admin) and (event.context != external)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == external)

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate AnalystOrPartner(event):
    (event.role == analyst) or (event.context == partner)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when (UnknownRisk(event)) and (not (UnknownExternal(event)))
    then prefer label = review priority 90

rule TrustedCase:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = review

rule ExportExternalCase:
    when (event.action == export) and (event.context == external)
    then prefer label = allow priority 70

rule AnalystOrPartnerCase:
    when AnalystOrPartner(event)
    then prefer label = review priority 40

rule OtherRoleCase:
    when ((event.role != admin) and (event.role != analyst)) and (event.context != partner)
    then prefer label = deny priority 1

rule AdminExternalCase:
    when (event.role == admin) and (event.context == external)
    then prefer label = deny priority 1

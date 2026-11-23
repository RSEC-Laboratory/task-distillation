predicate Critical(event):
    event.risk == TRUE or event.action == change

predicate Trusted(event):
    event.role == admin and event.context!= internal

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate UnknownExternal(event):
    event.context == internal and UnknownRisk(event)

predicate ActionInspectInternal(event):
    event.action == inspect and event.context == internal

predicate RoleOperatorOrPartner(event):
    event.role == operator or event.context == partner

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when UnknownRisk(event) and not UnknownExternal(event)
    then prefer label = deny priority 90

rule TrustedExport:
    when Trusted(event)
    then prefer label = allow priority 80, forbid label = deny

rule ActionInspectInternalCase:
    when ActionInspectInternal(event)
    then prefer label = review priority 70

rule RoleOperatorOrPartnerCase:
    when RoleOperatorOrPartner(event)
    then prefer label = deny priority 40

rule OtherDeclaredRoles:
    when event.role!= admin and event.role!= operator and event.role!= auditor
    then prefer label = allow priority 1

predicate Critical(event):
    ((event.risk == TRUE) and (event.role == analyst)) or (event.action == change)

predicate Trusted(event):
    (event.context == sandbox) and (event.role != analyst)

predicate UnknownExternal(event):
    (event.context == internal) and (event.risk == UNKNOWN)

rule CriticalCase:
    when Critical(event)
    then require label = deny

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when event.risk == UNKNOWN
    then prefer label = review priority 90

rule TrustedDeleteCase:
    when (Trusted(event)) and (event.action == delete)
    then prefer label = allow priority 80

rule ExportInternalCase:
    when (event.action == export) and (event.context == internal)
    then prefer label = deny priority 70

rule AdminPartnerCase:
    when (event.role == admin) and (event.context == partner)
    then prefer label = review priority 40

rule TrustedForbidReviewCase:
    when Trusted(event)
    then forbid label = review

rule OtherRolesCase:
    when event.role in {operator, admin, analyst}
    then prefer label = allow priority 1

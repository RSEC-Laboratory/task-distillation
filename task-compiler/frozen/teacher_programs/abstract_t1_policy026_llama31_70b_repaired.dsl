predicate Critical(event):
    (event.risk == TRUE) or ((event.context == sandbox) and (event.action == delete))

predicate Trusted(event):
    ((event.role == admin) or (event.role == analyst)) and (event.action != delete)

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == sandbox)

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedInspect:
    when (Trusted(event)) and (event.action == inspect)
    then prefer label = allow priority 80

rule SandboxApprove:
    when (event.context == sandbox) and (event.action == approve)
    then prefer label = review priority 70

rule AnalystOrInternal:
    when (event.role == analyst) or (event.context == internal)
    then prefer label = deny priority 40

rule TrustedForbidDeny:
    when Trusted(event)
    then forbid label = deny

rule OtherRoles:
    when (event.role == admin) or (event.role == auditor)
    then prefer label = allow priority 1

rule AnalystAllow:
    when (event.role == analyst) and (event.action != delete)
    then prefer label = allow priority 1

rule DefaultAllow:
    when not (((event.role == admin) or (event.role == analyst)) or (event.role == auditor))
    then prefer label = allow priority 1

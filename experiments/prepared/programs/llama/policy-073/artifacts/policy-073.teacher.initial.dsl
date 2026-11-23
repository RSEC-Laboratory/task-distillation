predicate Critical(event):
    event.risk == TRUE and event.context == sandbox or event.action == approve

predicate Trusted(event):
    event.role == auditor or event.context == partner and event.risk == FALSE

predicate UnknownExternal(event):
    event.risk == UNKNOWN and event.context == sandbox

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate GuestExternal(event):
    event.role == guest and event.context == external

predicate DeleteSandbox(event):
    event.action == delete and event.context == sandbox

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when UnknownRisk(event) and not UnknownExternal(event)
    then prefer label = deny priority 90

rule TrustedCase:
    when Trusted(event)
    then prefer label = allow priority 80, forbid label = deny

rule DeleteSandboxCase:
    when DeleteSandbox(event)
    then prefer label = review priority 70

rule GuestExternalCase:
    when GuestExternal(event)
    then prefer label = deny priority 40

rule OtherRoles:
    when event.role in {auditor, guest, analyst}
    then prefer label = allow priority 1
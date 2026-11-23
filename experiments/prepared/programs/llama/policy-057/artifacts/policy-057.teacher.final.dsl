predicate Critical(event):
    ((event.context == external) and (event.risk == TRUE)) or (event.action == change)

predicate Trusted(event):
    (event.role == auditor) or ((event.context == sandbox) and (event.risk == FALSE))

predicate UnknownRisk(event):
    event.risk == UNKNOWN

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when (UnknownRisk(event)) and (event.context == external)
    then abstain

rule UnknownRiskOtherwise:
    when (UnknownRisk(event)) and (not (event.context == external))
    then prefer label = allow priority 90

rule TrustedInspect:
    when (Trusted(event)) and (event.action == inspect)
    then prefer label = deny priority 80, forbid label = allow

rule DeleteExternal:
    when (event.action == delete) and (event.context == external)
    then prefer label = review priority 70

rule GuestInternal:
    when (event.role == guest) and (event.context == internal)
    then prefer label = allow priority 40

rule TrustedDeny:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = allow

rule OtherRoles:
    when (event.role != auditor) and (event.role != guest)
    then prefer label = deny priority 1

rule GuestSandbox:
    when (event.role == guest) and (event.context == sandbox)
    then prefer label = deny priority 1

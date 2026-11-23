predicate Critical(event):
    (event.risk == TRUE) or (event.action == delete)

predicate Trusted(event):
    (event.role == operator) and (event.context != external)

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate UnknownExternal(event):
    (event.risk == UNKNOWN) and (event.context == external)

predicate GuestOrSandbox(event):
    (event.role == guest) or (event.context == sandbox)

predicate OtherDeclaredRoles(event):
    ((event.role != operator) and (event.role != guest)) and (event.role != analyst)

rule CriticalCase:
    when Critical(event)
    then require label = allow

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when (UnknownRisk(event)) and (not (UnknownExternal(event)))
    then prefer label = review priority 90

rule TrustedApprove:
    when (Trusted(event)) and (event.action == approve)
    then prefer label = deny priority 80, forbid label = review

rule ExportInExternal:
    when (event.action == export) and (event.context == external)
    then prefer label = allow priority 70

rule GuestOrSandboxCase:
    when GuestOrSandbox(event)
    then prefer label = review priority 40

rule OtherRoles:
    when OtherDeclaredRoles(event)
    then prefer label = deny priority 1

rule DefaultDeny:
    when ((not (Trusted(event))) and (event.role != guest)) and (event.context != sandbox)
    then prefer label = deny priority 1

rule SourceCounterexamplePatch1:
    when (((event.role == operator) and (event.action == export)) and (event.context == sandbox)) and (event.risk == UNKNOWN)
    then require label = deny

rule SourceCounterexamplePatch2:
    when (((event.role == operator) and (event.action == export)) and (event.context == internal)) and (event.risk == UNKNOWN)
    then require label = deny

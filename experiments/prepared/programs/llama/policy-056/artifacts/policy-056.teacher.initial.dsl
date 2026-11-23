predicate Critical(event):
    event.risk == TRUE or event.action == export

predicate Trusted(event):
    event.role == admin and event.context != internal

predicate UnknownRisk(event):
    event.risk == UNKNOWN

predicate UnknownExternal(event):
    event.context == internal and UnknownRisk(event)

predicate ActionApprove(event):
    event.action == approve and event.context == internal

predicate RoleOperator(event):
    event.role == operator

predicate ContextSandbox(event):
    event.context == sandbox

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternalCase:
    when UnknownExternal(event)
    then abstain

rule UnknownRiskCase:
    when UnknownRisk(event) and not UnknownExternal(event)
    then prefer label = allow priority 90

rule TrustedCase:
    when Trusted(event)
    then prefer label = deny priority 80, forbid label = allow

rule ActionApproveCase:
    when ActionApprove(event)
    then prefer label = review priority 70

rule RoleOperatorCase:
    when RoleOperator(event) or ContextSandbox(event)
    then prefer label = allow priority 40

rule OtherRolesCase:
    when event.role in {admin, auditor}
    then prefer label = deny priority 1
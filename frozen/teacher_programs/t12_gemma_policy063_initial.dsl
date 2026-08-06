predicate Critical(event):
    event.risk == TRUE and event.role == admin or event.action == inspect

rule CriticalCase:
    when Critical(event)
    then require label = review

rule UnknownExternal:
    when event.risk == UNKNOWN and event.context == external
    then abstain
rule UnknownRisk:
    when event.risk == UNKNOWN
    then prefer label = deny priority 90

rule TrustedExport:
    when event.context == partner and event.role != admin
    then prefer label = allow priority 80
rule TrustedExportForbidDeny:
    when (event.context == partner and event.role != admin)
    then forbid label = deny

rule ActionChangeInternal:
    when event.action == change and event.context == internal
    then prefer label = review priority 70

rule GuestExternal:
    when event.role == guest and event.context == external
    then prefer label = deny priority 40

rule DefaultAllow:
    when event.role == auditor or event.role != auditor
    then prefer label = allow priority 1

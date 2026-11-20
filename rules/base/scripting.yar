rule Suspicious_PowerShell_Launcher
{
  meta:
    description = "Encoded or download-cradle PowerShell command lines"
  strings:
    $p0 = "powershell" ascii wide nocase
    $p1 = "-enc" ascii wide nocase
    $p2 = "-nop" ascii wide nocase
    $p3 = "IEX" ascii wide
    $p4 = "DownloadString" ascii wide nocase
    $p5 = "FromBase64String" ascii wide nocase
    $p6 = "-w hidden" ascii wide nocase
  condition:
    $p0 and 2 of ($p1, $p2, $p3, $p4, $p5, $p6)
}

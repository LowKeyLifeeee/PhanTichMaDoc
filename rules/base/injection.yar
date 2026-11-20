rule Process_Injection_API_Combo
{
  meta:
    description = "Imports or strings for classic remote-process injection"
  strings:
    $a0 = "VirtualAllocEx"
    $a1 = "WriteProcessMemory"
    $a2 = "CreateRemoteThread"
    $a3 = "NtUnmapViewOfSection"
    $a4 = "SetThreadContext"
    $a5 = "QueueUserAPC"
    $a6 = "NtWriteVirtualMemory"
  condition:
    uint16(0) == 0x5A4D and 3 of them
}

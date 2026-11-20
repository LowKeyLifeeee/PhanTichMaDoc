rule UPX_Packed
{
  meta:
    description = "UPX section names or stub string in a PE file"
  strings:
    $s0 = "UPX0"
    $s1 = "UPX1"
    $s2 = "UPX!"
  condition:
    uint16(0) == 0x5A4D and 2 of them
}

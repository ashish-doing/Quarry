$prefix = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like "10.*" -and $_.IPAddress -notlike "*.1" }).IPAddress -replace '\.\d+$', ''
Write-Host "Scanning $prefix.0/24 ..."
1..254 | ForEach-Object { Ping -n 1 -w 100 "$prefix.$_" | Out-Null }
arp -a | Select-String $prefix
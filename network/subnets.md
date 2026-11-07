# Task 5 — Subnetting and Network Access Control

Source block: `10.20.0.0/16`. Rule used throughout: usable hosts = 2^(32 − prefix) − 2
(network and broadcast addresses are not usable hosts).

## Working

Allocate largest requirement first, each subnet placed immediately after the
previous one, using the smallest prefix that fits:

- **app — 1,000 hosts:** need 2^10 − 2 = 1,022 ≥ 1,000, so 10 host bits → **/22**.
  Starts at 10.20.0.0, covers 10.20.0.0–10.20.3.255.
- **public — 250 hosts:** need 2^8 − 2 = 254 ≥ 250, so 8 host bits → **/24**.
  Starts right after app, at 10.20.4.0, covers 10.20.4.0–10.20.4.255.
- **data — 100 hosts:** need 2^7 − 2 = 126 ≥ 100, so 7 host bits → **/25**.
  Starts right after public, at 10.20.5.0, covers 10.20.5.0–10.20.5.127.
- **management — 20 hosts:** need 2^5 − 2 = 30 ≥ 20, so 5 host bits → **/27**.
  Starts right after data, at 10.20.5.128, covers 10.20.5.128–10.20.5.159.

No two subnets overlap, and nothing is wasted beyond the rounding up to the
next power of two (e.g. app uses 1,024 addresses for 1,000 hosts; a /23 would
give only 510 usable hosts and would not fit).

## Subnet table

| Subnet | Network address | Prefix | Usable host range | Broadcast address | Usable hosts |
| --- | --- | --- | --- | --- | --- |
| app | 10.20.0.0 | /22 | 10.20.0.1 – 10.20.3.254 | 10.20.3.255 | 1,022 |
| public | 10.20.4.0 | /24 | 10.20.4.1 – 10.20.4.254 | 10.20.4.255 | 254 |
| data | 10.20.5.0 | /25 | 10.20.5.1 – 10.20.5.126 | 10.20.5.127 | 126 |
| management | 10.20.5.128 | /27 | 10.20.5.129 – 10.20.5.158 | 10.20.5.159 | 30 |

## NACL for the public subnet — `network/nacl.json`

The load balancer lives in the public subnet (10.20.4.0/24). Rules are
checked lowest number first; the first match decides, and anything with no
match is denied.

**Inbound**

| Rule | Action | Protocol / port | Source |
| --- | --- | --- | --- |
| 100 | deny | all | 198.51.100.0/24 (abusive range) |
| 110 | allow | TCP 80 | 0.0.0.0/0 |
| 120 | allow | TCP 443 | 0.0.0.0/0 |
| 130 | allow | TCP 22 | 10.20.5.128/27 (management only) |
| 140 | allow | TCP 1024–65535 | 10.20.0.0/22 (replies from the app subnet) |

**Outbound**

| Rule | Action | Protocol / port | Destination |
| --- | --- | --- | --- |
| 100 | allow | TCP 8000 | 10.20.0.0/22 (app subnet) |
| 110 | deny | TCP 8000 | 0.0.0.0/0 (8000 anywhere else) |
| 120 | allow | TCP 1024–65535 | 0.0.0.0/0 (replies back to clients) |

## Why a stateless NACL needs separate reply rules, and why rule order matters

A NACL is stateless: it does not remember connections. When the load
balancer sends a request to the app service on TCP 8000, the app's reply
comes back as a brand-new inbound packet — source in the app subnet,
destination port in the client's ephemeral range (1024–65535). Nothing in
the NACL remembers that the outbound request happened, so without inbound
rule 140 that reply would hit the default deny and the connection would
never complete, even though the outbound request was allowed. The same
applies in reverse: a client's request arriving on TCP 80/443 is allowed
inbound, but its reply leaves as a new outbound packet to an ephemeral
port, so outbound rule 120 is needed for it. Each direction of every
connection therefore needs its own explicit rule.

Rule order matters because the first matching rule decides. The abusive
range 198.51.100.0/24 must be denied by rule 100, *before* the general
web allows in rules 110 (TCP 80) and 120 (TCP 443). If the deny were
numbered higher than those allows, a packet from 198.51.100.9 to TCP 443
would match the allow first and be let through, and the deny would never
be reached — the block would be dead. Putting the specific deny first is
what makes it override the general allow.

## Verification




```
python3 student_kit/tools/nacl_sim.py network/nacl.json student_kit/network/packets_sample.jsonl --check
```

Result: **6 / 6 correct (100%)**.

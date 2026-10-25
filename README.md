# boxx1 — box on-demand (VPS + badan baru)

GitHub Actions runner dipakai sebagai mesin: **4 vCPU / 16 GB RAM / Ubuntu 24.04**, sesi maks **6 jam** per run.
User login: **`runzzx1`** (sudo NOPASSWD). Password **acak, digenerate tiap run**.

## Cara pakai

1. Tab **Actions** → workflow **vps** → **Run workflow**
2. Isi `hours` (default 5) — `playwright: yes` kalau butuh chromium
3. Tunggu ±60 detik, buka **Summary** run-nya. Ada 3 jalur koneksi:

| Jalur | Host | Port | Catatan |
|---|---|---|---|
| **Tailscale** | `100.x.x.x` | 22 | stabil, gak ganti selama sesi — **pakai ini** |
| pinggy | `xxx.run.pinggy-free.link` | random | ganti tiap ±60 menit |
| web terminal | `https://xxx.trycloudflare.com` | — | buka di browser, login user/pass sama |

Kredensial juga dikirim ke ntfy: `https://ntfy.sh/boxx1-bd30af96b8c8`

## Auto-revive

`keepalive.yml` jalan tiap 20 menit; kalau gak ada run aktif, dia dispatch `vps.yml` sendiri.
Butuh secret `STATE_TOKEN` (PAT dengan scope `repo` + `workflow`). Kalau kosong, keepalive skip.

## Secret yang dipakai

- `TS_AUTHKEY` — Tailscale auth key (reusable, tailnet `setyoanugrah2@`)
- `NTFY_TOPIC` — topik ntfy buat push kredensial
- `STATE_TOKEN` — PAT buat keepalive

## Akses Tailscale dari HP

Install Tailscale di HP, login akun yang sama → `ssh runzzx1@100.93.33.49` (IP cek di Summary).
Dari Termux tanpa root: jalankan `tailscaled --tun=userspace-networking` lalu
`ssh -o ProxyCommand="tailscale nc %h %p" runzzx1@<ip>`.

## Catatan

- Job mati otomatis di 6 jam. `hours` > 5 bakal ke-cut, jadi default 5.
- Repo public → menit Actions unlimited.
- Tailscale hostname selalu `runzzx1`; kalau run lama masih hidup, IP-nya bisa tabrakan — matikan run lama dulu.

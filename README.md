# boxx1 — VPS on-demand + Minecraft Bedrock server

GitHub Actions runner dipakai sebagai mesin: **4 vCPU / 16 GB RAM / Ubuntu 24.04**, sesi maks **6 jam** per run.
User login: **`runzzx1`** (sudo NOPASSWD). Password **acak, digenerate tiap run**.

## Cara pakai

1. Tab **Actions** → workflow **vps** → **Run workflow**
2. Isi `hours` (default 5), `minecraft` = `yes` (default), `playwright` = `yes` kalau butuh chromium
3. Tunggu ±3 menit, buka **Summary** run-nya. Kredensial juga di-push ke ntfy.

## Minecraft Bedrock

| Item | Nilai |
|---|---|
| Alamat | IP Tailscale box (`100.x.x.x`) |
| Port | `19132` |
| Versi | latest otomatis (sekarang 1.26.52.3) |
| Mode | survival, normal, max 10, cheats on |

Di Minecraft: **Play → Servers → Add Server** → isi alamat + port `19132`.
HP harus connect ke Tailscale (akun `setyoanugrah2@`) dulu — Bedrock itu **UDP**, jadi cloudflared/pinggy
(TCP-only) **tidak bisa** dipakai buat game-nya.

### Kenapa `transport=raknet`

Bedrock 1.26+ default-nya `transport=nethernet` (WebRTC) yang pakai port acak dan tidak bisa
di-port-forward / di-tunnel UDP biasa. `scripts/mc_setup.py` memaksa `raknet` supaya bind ke 19132.

### Kelola server (lewat SSH)

```bash
~/mc/mcctl.sh status     # pid, port, log terakhir
~/mc/mcctl.sh start
~/mc/mcctl.sh stop
~/mc/srv/server.properties   # config (butuh restart)
~/mc/srv.log                 # log
```

Semua di `~/mc` milik `runzzx1` — bisa diutak-atik tanpa sudo.

### World persistence

`mcworld` (repo privat) nyimpen `worlds/` dan di-backup tiap 10 menit selama sesi hidup.
Run berikutnya otomatis restore dari situ. Kalau mau world fresh, kosongkan folder `worlds/` di repo itu.

## Koneksi SSH

| Jalur | Host | Port | Catatan |
|---|---|---|---|
| **Tailscale** | `100.x.x.x` | 22 | stabil selama sesi — pakai ini |
| pinggy | `xxx.run.pinggy-free.link` | random | ganti tiap ±60 menit |
| web terminal | `https://xxx.trycloudflare.com` | — | buka browser, login user/pass sama |

ntfy topic: `https://ntfy.sh/boxx1-bd30af96b8c8`

## Auto-revive

`keepalive.yml` jalan tiap 20 menit; kalau tidak ada run aktif, dia dispatch `vps.yml` sendiri.
Butuh secret `STATE_TOKEN` (PAT scope `repo` + `workflow`).

## Secret

- `TS_AUTHKEY` — Tailscale auth key (tailnet `setyoanugrah2@`)
- `NTFY_TOPIC` — topik ntfy
- `STATE_TOKEN` — PAT buat keepalive + world backup

## Struktur

```
.github/workflows/vps.yml      # boot box + MC + publish + backup loop + hold
.github/workflows/keepalive.yml
scripts/mc_version.py          # resolve URL server Bedrock terbaru
scripts/mc_setup.py            # patch server.properties (paksa raknet)
scripts/mc_bootstrap.sh        # unduh, restore world, config, start
scripts/mcctl.sh               # start/stop/status
scripts/mc_backup.sh           # backup worlds/ ke repo privat
scripts/mcping.py              # RakNet ping (self-test)
```

## Akses Tailscale dari HP

Install Tailscale di HP, login akun yang sama → `ssh runzzx1@<ip-dari-summary>`.
Dari Termux tanpa root: `tailscaled --tun=userspace-networking` lalu
`ssh -o ProxyCommand="tailscale nc %h %p" runzzx1@<ip>`.

## Catatan

- Job mati otomatis di 6 jam. `hours` > 5 bakal ke-cut, jadi default 5.
- Repo public → menit Actions unlimited.
- Tailscale hostname `runzzx1`; kalau run lama masih hidup, IP-nya jadi `runzzx1-2` dst.

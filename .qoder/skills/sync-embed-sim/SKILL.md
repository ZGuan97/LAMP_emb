---
name: sync-embed-sim
description: Sync the local embed_sim repository to the Tianjin supercomputer at ~/dmet/embed_sim/. Uses rsync with --delete to replace the old version. Use when the user asks to sync embed_sim to 天津、超算, or push the latest code to the remote server.
---

# Sync embed_sim to Tianjin Server

Sync `/Users/zhebin/work/embed_sim/` to `~/dmet/embed_sim/` on `tj1.chinahpc.com`.

Connection details (host, port, user, key) are in the `df-tj-remote` skill.

## Sync Command

```bash
rsync -avz --delete --progress \
  -e "ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb" \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.ipynb_checkpoints/' \
  /Users/zhebin/work/embed_sim/ \
  df_iopcas_gzb@tj1.chinahpc.com:~/dmet/embed_sim/
```

- `--delete` removes stale files on the remote that no longer exist locally.
- The trailing slash on local path and remote path is important for directory-to-directory sync.
- `CoSPh4/` in `~/dmet/` is **not** part of the repo and will not be affected.

## Verify

After sync, check remote state:

```bash
ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb df_iopcas_gzb@tj1.chinahpc.com \
  'cd ~/dmet/embed_sim && git log --oneline -3 && echo "---" && git branch && echo "---" && git status --short'
```

## Cleanup

If `~/embed_sim/` was created by mistake (wrong target path), delete it using `del`:

```bash
ssh -p 1014 -i ~/.ssh/df_tj_iopcas_gzb df_iopcas_gzb@tj1.chinahpc.com \
  '/data/home/df_iopcas_gzb/bin/del ~/embed_sim'
```

Never use `rm -rf` on the remote server. `del` moves files to `~/trash/<timestamp>/` for recovery.

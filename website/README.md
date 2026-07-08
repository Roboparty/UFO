# UFO Demo Gallery

Static website for presenting UFO policies, tracking rollouts, reward prompts, and goal-conditioned behavior.

## Preview

From the repository root:

```bash
tools/sync_website_media.sh
python3 -m http.server 18180
```

Open:

```text
http://localhost:18180/website/
```

If the server is remote:

```bash
ssh -p 51822 -L 18180:localhost:18180 xue@223.167.85.187
```

Then open the same local URL.

## Media

The gallery expects small mp4 files under `website/public/media/`. These videos are generated experiment artifacts and are intentionally ignored by Git. Run `tools/sync_website_media.sh` on the training machine to populate them from `/data/xue/bfmzero-mjlab/runs`.

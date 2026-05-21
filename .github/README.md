# GitHub metadata

## Pinned repository

Pin **3DGS-VLM** on your profile — recruiters often open only the pinned repo.

## Profile README (optional)

Create a public repo named `<your-username>/<your-username>` and add a short `README.md`:

```markdown
### Hi
- 🔭 Working on **3DGS + VLM** spatial referring ([3DGS-VLM](https://github.com/<user>/3DGS-VLM))
- 🧰 Python · PyTorch · 3DGS · multimodal fine-tuning
```

## Before each push

1. `git status` — no `*.pth`, `*.safetensors`, `test2/runs/`, `training_data/**/image/`
2. README Highlights match `docs/results_2d_eval.json` / `docs/depth_compare_batch.json`
3. `demo/pipeline.png` / `compare_2d.png` — optional; keep under ~2 MB each

## Thin-repo maintenance

| Event | Action |
|-------|--------|
| New bridge script | `git add bridge/` |
| Upstream version bump | Update `docs/UPSTREAM_SETUP.md` commit hash / branch note only |
| New RoboRefer patch | Extend `patches/roborefer/INTEGRATION.md` |
| New result run | Update public JSON under `docs/`, not raw run folders |

## Issues / Discussions

Solo research prototype — Issues can stay **disabled** or label as *no maintenance*. Point readers to `docs/UPSTREAM_SETUP.md`.

## Release tags (optional)

`v0.1.0` — tag when resume PDF + demo images are frozen; no need for frequent releases.

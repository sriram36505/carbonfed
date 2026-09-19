# Pushing this to GitHub and minting a DOI

## 1. Create the repository

```bash
cd carbonfed
git init
git add .
git commit -m "CarbonFed: reference implementation"
git branch -M main
git remote add origin https://github.com/<your-username>/carbonfed.git
git push -u origin main
```

Make the repository **public** — a private repo cannot be cited.

## 2. Mint a DOI with Zenodo

1. Sign in at <https://zenodo.org> with your GitHub account.
2. Go to **Settings -> GitHub**, find `carbonfed`, and flip the toggle **on**.
3. Back on GitHub: **Releases -> Create a new release**, tag `v1.0.0`, publish.
4. Zenodo archives the release and issues a DOI within a minute or two.

## 3. Use the DOI in the submission

Paste the DOI URL (`https://doi.org/10.5281/zenodo.XXXXXXX`) into the
"Link to data repository" field. Set **Repository name** to `Zenodo` and
**Source of data** to `Original data`.

## 4. Update the manuscript

Replace the Data availability statement with the deposited link, for example:

> The simulation environment, agent implementation, configuration files and
> analysis scripts used in this study, together with fixed random seeds, are
> openly available at https://doi.org/10.5281/zenodo.XXXXXXX.

## Before you push

- Replace `<your-username>` in `README.md` and `CITATION.cff`.
- Confirm the author list and affiliations in `LICENSE` and `CITATION.cff`.
- Check that `data/raw/` contains no licensed or redistribution-restricted traces.

# Mining Example

This folder contains a simple single-paper PubMed/PMC text-mining workflow.

## Install

You need Python plus:

```bash
pip install biopython tqdm
```

Optional, only if you use `--plots`:

```bash
pip install matplotlib
```

## Run One Paper

From this folder:

```bash
python3 download.py --pmid 36979433 --outdir single_article_pmid_36979433/pmc_articles_2023_pmid_36979433
python3 text_mining.py -i single_article_pmid_36979433 -o output_pmid_36979433
```

Results will be here:

```text
output_pmid_36979433/results_2023.tsv
output_pmid_36979433/summary_2023.json
```

## Another Example

```bash
python3 download.py --pmid 40425816 --outdir single_article_pmid_40425816/pmc_articles_2025_pmid_40425816
python3 text_mining.py -i single_article_pmid_40425816 -o output_pmid_40425816
```

## Output Columns

The TSV includes the detected evidence words/phrases:

```text
algorithms_found
params_found
justification_matches
evaluation_matches
tuning_matches
missing_reporting_signals
```

`missing_reporting_signals = 0` means the script found algorithm, parameter, justification, evaluation, and tuning signals.

## Optional NCBI Settings

If you have an NCBI API key or want to set your email:

```bash
export NCBI_EMAIL="your.email@example.com"
export NCBI_API_KEY="your_api_key"
```

# RapCluster web page: user guide

This guide explains every part of the RapCluster web page, from top to bottom. Most labels on the page also have a
small **?** marker: hover over it (or tap it, or focus it with the Tab key) to see a short explanation. On buttons
and tabs, the same explanation appears as the button's tooltip.

The usual workflow is:

1. upload a table (or load an example),
2. choose the preprocessing,
3. choose a dimensionality reduction (optional),
4. choose a clustering algorithm and its parameters,
5. run, then look at the results, compare runs and download what you need.

Every run is saved in the session, so you can try many settings on the same upload and compare them. All data and
runs are deleted when you close the page.

---

## 1. Upload Data

| Field | What it does |
|---|---|
| **Select File** | CSV, TSV, XLSX or XLS, up to 50 MB. One row per item (protein, gene, sample, …), one column per measurement. |
| **Or load example data** | Loads one of the bundled datasets and fills in the name column, data start column and suggested preprocessing. |
| **Sheet** | Only for Excel files: the worksheet that holds the data. |
| **Name of Node Column** | The column with the identifier of each row. It labels points, tables and downloads; it is not clustered. |
| **Data Start Column Index (0-based)** | Position of the first numeric column, counting from 0. This column and every later one are used as data; earlier columns (names, annotations, known classes) are ignored. |

Example datasets:

| Example | Size | Good for |
|---|---|---|
| Protein profiles | 3000 × 120 | the original RapCluster use case (fractionation profiles) |
| Time-course profiles | 1200 × 24 | six expression patterns over time |
| Gaussian blobs | 600 × 10 | five well-separated groups; a sanity check |
| Uneven density with noise | 700 × 8 | DBSCAN, HDBSCAN and OPTICS |
| Iris | 150 × 4 | the classic small dataset |

The examples have a `Class` column with the known groups. It lies before the data start column, so it is not used
for clustering, but you can compare the result with it.

## 2. Preprocessing

The steps run in the order listed and are saved with every run. The defaults reproduce RapCluster's original
pipeline.

| Setting | Options and meaning |
|---|---|
| **Remove rows with more than … % missing values** | Drops rows with too many empty cells. 100 keeps every row. |
| **Fill remaining missing values with** | 0 (default), column minimum, half of the column minimum, or column median. For proteomics data, missing often means "below detection", so the minimum or half-minimum can be more realistic than 0. |
| **Transformation** | log2(x + 1) (default), log10(x + 1), signed log (works with negative values), or none. A log compresses very large values so they do not dominate. |
| **Scaling** | Per-row z-score (default) or per-row min–max compare the *shape* of each profile and ignore its level; suited to fractionation or time-course data. Per-column z-score, robust (median/IQR) or min–max put every measurement on the same scale; suited to columns of different kinds (e.g. iris). None clusters the transformed values directly. |
| **Remove all-zero rows** | Rows without any signal (on by default). |
| **Remove low-variance columns** | Removes columns whose standard deviation after the transformation is at or below the threshold (0 = constant columns only). |
| **Remove highly correlated columns** | Removes a column when its correlation with an earlier, kept column is above the threshold. Not recommended for profile data, where neighbouring columns are correlated by nature. |
| **Explore data with these settings** | Opens the **Exploratory Analysis** panel (see below) without clustering. |

### Exploratory Analysis

Shows the data after the chosen preprocessing: missing values per row and column, PCA variance explained (how many
dimensions carry most of the structure), value distributions before and after scaling, a correlation heatmap of the
columns, and rows that lie far from the overall median profile.

## 3. Dimensionality Reduction

Optional. Maps the data to fewer dimensions before clustering. Clustering uses **all** chosen dimensions; the plots
show the first two. **Learn More** opens a longer guide.

| Method | Short description | Main parameters |
|---|---|---|
| **None** | Clustering uses all features. The scatter plot shows a PCA projection for display only. | – |
| **PCA** | Linear projection onto the directions of largest variance. Fast and deterministic. | `n_components`, `svd_solver` |
| **UMAP** | Non-linear; keeps neighbours together and much of the global layout. Good input for density-based clustering. | `n_components`, `n_neighbors` (local vs. global), `min_dist` (how tightly points pack), `metric` |
| **TSNE** | Non-linear; keeps close neighbours together and gives clearly separated groups, but group sizes and distances in the plot have little meaning. 2 or 3 dimensions, slow on large data. | `perplexity`, `learning_rate`, `n_iter`, `metric` |

`random_state` is the random seed: the same seed gives the same result.

## 4. Clustering Algorithm

**Use Default Parameters** runs the algorithm with its defaults. Untick it to change single parameters; each one has a
**?** with its meaning, default and allowed range. **Learn More** opens the algorithm guide.

| Algorithm | Idea | You set | Noise label | Most important parameters |
|---|---|---|---|---|
| **KMeans** | Rows close to the mean of their group | number of clusters | no | `n_clusters` |
| **MiniBatchKMeans** | KMeans on small random batches; for large data | number of clusters | no | `n_clusters` |
| **BisectingKMeans** | Starts with one cluster and keeps splitting one in two | number of clusters | no | `n_clusters`, `bisecting_strategy` |
| **AgglomerativeClustering** | Merges the closest clusters step by step | number of clusters | no | `n_clusters`, `linkage` |
| **BIRCH** | Compresses the data into a tree of subclusters; for large data | number of clusters (or None) | no | `threshold`, `n_clusters` |
| **SpectralClustering** | Clusters a similarity graph; finds curved shapes | number of clusters | no | `affinity`, `n_neighbors` |
| **GaussianMixture** | Mixture of Gaussian distributions; soft assignments | number of components | no | `n_components`, `covariance_type` |
| **BayesianGaussianMixture** | Gaussian mixture that leaves unneeded components empty | maximum number of components | no | `weight_concentration_prior`, `covariance_type` |
| **DBSCAN** | Connected dense regions | density (`eps`, `min_samples`) | yes | `eps`, `min_samples` |
| **HDBSCAN** | DBSCAN over all density levels; varying density | minimum cluster size | yes | `min_cluster_size`, `min_samples` |
| **OPTICS** | Density ordering, clusters cut at steep changes | density (`xi`, `min_samples`) | yes | `xi`, `min_samples` |
| **MeanShift** | Every density peak is a cluster | window size (`bandwidth`) | optional | `bandwidth` |
| **AffinityPropagation** | Rows elect exemplars; only for small data | preference | no | `preference`, `damping` |

Noise rows get the label **-1** and are shown in grey. They are not a cluster and are left out of the metrics.

## Check and run

- **Check data & settings** inspects the data and the chosen methods without clustering and opens the
  **Preflight Check** panel.
- **Run Clustering** starts the run as a background job. The preflight check runs first; if a method is not
  recommended for the data size, you are asked before the run starts. A progress bar shows the current stage, and the
  run can be cancelled.
- **Watch Youtube Tutorial** links to a video introduction.

### Preflight Check

- **Data checks**: rows and columns used, missing and non-numeric values, values that a log transform cannot handle,
  constant columns, all-zero rows, duplicate rows and names, strongly correlated columns, estimated result size, and
  whether the scaling fits the data.
- **Capacity guidance**: every algorithm, reduction and metric is rated *Recommended*, *May be slow* or *Not
  recommended* for this data size on the hosted server, with the reason (for example, memory that grows with the
  square of the number of rows).

Limits on the hosted server: 50 MB per upload, 10 minutes and 4 GB per run, 3 runs at the same time. Error messages
say which limit was reached and suggest what to change.

## Parameter Sweep

Runs the selected algorithm several times with different values of one or two parameters. Preprocessing, reduction and
all other parameters stay fixed, so the trials are comparable.

- **Parameter 1 / 2**: for numbers give *from*, *to* and *step*; for options tick the values. *also None* adds the
  "automatic" value where a parameter allows it.
- **Max. trials** (up to 30): if there are more combinations, a random subset (fixed seed) is tried.
- **Screen on a subsample first**: every trial runs on a share of the rows, and only the best *k* (by the chosen
  metric) are repeated on all rows.
- **Results**: a metric-versus-parameter curve (or a heatmap for two parameters), a table of the trials with **View**
  buttons, failed trials with their error, the best trial for the chosen metric and constraints, and the sweep
  configuration as JSON. Every finished trial is a normal run with a *sweep* tag in Run History.

## Run History

All runs of the session, newest first: run number, time, dataset, reduction, algorithm, number of clusters, noise,
Silhouette, runtime, peak memory and status. **View** shows a run and puts its settings back into the form;
**Details** lists every setting; **Labels TSV** and **ZIP** download the results. Failed runs show their error and a
**Load settings** button. **Hide / Show** collapses the table.

## Run Leaderboard

Ranks the runs and names the best one *for the chosen criterion under the chosen constraints*. A different criterion
can favour another run; no single score finds the best clustering for every purpose.

- **Rank by**: Silhouette, Calinski–Harabasz, DBCV, stability (higher is better), Davies–Bouldin, runtime (lower is
  better), or a **Pareto ranking** over several objectives (front F1 = runs that no other run beats on every chosen
  objective).
- **Constraints**: number of clusters, maximum noise %, minimum cluster size, maximum runtime, maximum size imbalance
  (largest / smallest cluster) and minimum stability. Excluded runs can be shown with the reason.
- **"Recommended" badge**: given only when quality (Silhouette ≥ 0.25 or a DBCV threshold) *and* stability (mean ARI ≥
  0.75) are both reached. The thresholds can be changed. Runs without a stability analysis cannot get the badge;
  **Compute stability** runs it for them.
- **Pin**: keeps the current best run in view while you try other settings.
- A stability-versus-quality plot shows all runs at once.

## Compare Runs

Puts two runs (A and B) side by side, on all rows:

- each run's settings, scatter plot, cluster sizes, profile heatmap, metrics, runtime, memory and warnings,
- **ARI** and **NMI** between the two clusterings (1 = identical),
- **parameter differences**, including preprocessing,
- **how samples move**: a contingency matrix or a Sankey diagram,
- **cluster matching by overlap** (cluster numbers have no meaning across runs, so clusters are paired by their shared
  rows),
- **samples that changed cluster**, downloadable as a list.

*Color Run B by matched Run A clusters* gives matched clusters the same colour.

## Clustering Results

The panel for the selected run.

- **Download All (ZIP)**: everything about the run (see below).
- A note says how many dimensions were clustered and what the scatter plot shows. For large data, plots show a sample
  of up to 5000 rows that includes every cluster; sizes, summaries and downloads use all rows.

### Metrics

| Metric | Meaning | Better |
|---|---|---|
| **Clusters Found** | Number of clusters, noise not counted | – |
| **Silhouette Score** | How much closer rows are to their own cluster than to the next one (−1 to 1). Computed on up to 5000 rows. | higher |
| **Calinski–Harabasz** | Spread between clusters relative to spread within them. No fixed scale. | higher |
| **Davies–Bouldin** | How similar each cluster is to its closest neighbour cluster. 0 is best. | lower |
| **DBCV** | Density-based validation; also fair to non-round clusters (−1 to 1). | higher |
| **BIC / AIC** | Fit of mixture models with a penalty for complexity; compare runs on the same data only. | lower |
| **Mean membership** | Average probability with which rows belong to their cluster (HDBSCAN, mixtures). | higher |
| **Components used** | Mixture components that received rows (BayesianGaussianMixture). | – |
| **Runtime** | Time for the whole run. | – |

### Visualization tabs

- **Scatter Plot**: every row as a point coloured by cluster. Hover for the name; drag to zoom. A caption template
  below the plot describes the settings for a figure legend.
- **Clusters Network**: one node per cluster at its centre; edges between clusters are thicker and darker when the
  clusters are closer. Click a cluster to open its sub-network.
- **Sub-network View**: the rows of one cluster, each connected to its nearest neighbours (choose 3–20). Draw a lasso
  around points to plot their raw profiles (log10). *Pan/Zoom* switches the mouse mode.
- **Heatmap**: preprocessed values as colours, either one summary row per cluster (median or mean) or the rows of one
  cluster (up to 1000). The optional dendrogram only orders the rows; it is not another clustering.
- **Characterization** (computed on all rows):
  - *Cluster summary*: size, share, medoid (the most typical member), representative members, and the most
    distinguishing features. Member lists can be copied or downloaded, e.g. for g:Profiler or STRING.
  - *Cluster profiles*: median per column with the 25–75% band.
  - *Features that distinguish a cluster*: Cliff's δ of the cluster against all other rows (+1 = always higher,
    −1 = always lower), as bars and a table, plus box plots of a chosen feature in every cluster.
  - *Differences between clusters per feature*: Kruskal–Wallis test with ε² effect size and Benjamini–Hochberg
    q-values. Descriptive only, because the clusters were built to differ.
  - *Outliers and noise*: rows far from their cluster's median profile (robust z-score), and the noise rows.
  - For HDBSCAN and mixtures, extra columns show mean membership, cluster persistence or mixture weights.
- **Stability**: re-clusters 10 random 80% subsamples (3–30) and, for algorithms with a seed, 5 runs with other seeds
  (0–10).
  - *Cluster stability*: mean Jaccard overlap of each cluster with its best match in the repeated solutions;
    ≥ 0.75 stable, 0.6–0.75 doubtful, < 0.6 unstable.
  - *Agreement between solutions*: ARI/NMI of the repeated solutions against the run and against each other.
  - *Row stability*: how often each row keeps its cluster, with a histogram and the least stable rows. The
    *consensus cluster* is the cluster a row was assigned to most often.
  - Row stability is added to Cluster Details, the labels file and the Characterization table.

### Method Details

A summary of the run's settings and a methods paragraph you can adapt for a paper. **Download Method Details** saves
it as text.

### Cluster Details

One table row per data row: name, cluster, plot coordinates, colour, and (when available) membership probability and
row stability. Click a column header to sort. The table can be hidden.

### Download All (ZIP)

| File | Content |
|---|---|
| `cluster_assignments.tsv` | Cluster of every row, with name, original row number and plot coordinates (plus membership and stability when available) |
| `cluster_sizes.tsv`, `cluster_centroids.tsv`, `cluster_medians.tsv` | Size, mean and median profile per cluster |
| `cluster_summary.tsv`, `cluster_profile_bands.tsv`, `distinguishing_features.tsv`, `cliffs_delta_matrix.tsv`, `feature_tests.tsv`, `outliers.tsv`, `noise_points.tsv`, `cluster_members/` | The characterization |
| `cluster_stability.tsv` | Stability per cluster (after a stability analysis) |
| `scatter`, `cluster_sizes`, `cluster_profiles` (`.png` / `.svg`) | Figures |
| `metrics.json`, `config.json`, `run_record.json`, `software_versions.json`, `warnings.txt` | Metrics, settings with seeds, full run record, package versions, warnings |
| `report.html` | Readable report with figures, characterization, stability, settings and methods text |
| `reproduce.py`, `preprocessing.py` | `python reproduce.py <data file>` repeats the run outside the web app |

---

## Glossary

- **ARI (adjusted Rand index)**: agreement of two clusterings over all pairs of rows, corrected for chance. 1 =
  identical, about 0 = random.
- **NMI (normalized mutual information)**: how much one clustering tells about the other, from 0 to 1.
- **Jaccard overlap**: shared rows divided by the rows in either of two clusters.
- **Cliff's δ**: effect size between two groups of values, from −1 to +1; |δ| above about 0.47 is a large effect.
- **Medoid**: the cluster member with the smallest total distance to the other members.
- **Noise (-1)**: rows that density-based algorithms do not assign to any cluster.
- **Robust z-score**: distance from the median divided by the scaled median absolute deviation; less affected by
  outliers than the usual z-score.
- **Seed (`random_state`)**: fixes the random choices of an algorithm so a run can be repeated exactly.

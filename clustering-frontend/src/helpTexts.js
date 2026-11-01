// Texts for the "?" hover help (HelpTip) and the method details. One place, so the explanations stay consistent
// between the form, the parameter sweep and the result sections.

export const ALGORITHM_HELP = {
  MiniBatchKMeans:
    'K-Means fitted on small random batches of rows instead of all rows in every step. Much faster and lighter on '
    + 'memory than KMeans, with almost the same result. You choose the number of clusters and every row is assigned '
    + '(no noise). Assumes compact, roughly round clusters. A good first choice for large datasets.',
  KMeans:
    'Splits the rows into k groups so that every row is as close as possible to the mean (centroid) of its group, '
    + 'i.e. it minimizes the variance within the clusters. You choose k and every row is assigned (no noise). Works '
    + 'best for compact, round clusters of similar size; struggles with elongated shapes or very different densities.',
  AffinityPropagation:
    'Rows exchange "messages" until some of them emerge as exemplars (representatives); every other row joins its '
    + 'most similar exemplar. The number of clusters is not set directly but follows from the preference. Time and '
    + 'memory grow with the square of the number of rows, so it is only suitable for small datasets.',
  MeanShift:
    'Moves a window over the data towards the regions of highest density; every density peak becomes a cluster. '
    + 'Finds the number of clusters by itself; the bandwidth (window size) decides how many peaks are found. '
    + 'Slow on large datasets.',
  SpectralClustering:
    'Builds a similarity graph between the rows, embeds the graph with the eigenvectors of its Laplacian and '
    + 'clusters that embedding. Can find curved or ring-shaped clusters that K-Means misses. You choose the number '
    + 'of clusters. Needs a lot of memory for many rows, especially with the rbf affinity.',
  AgglomerativeClustering:
    'Bottom-up hierarchical clustering: every row starts as its own cluster, and the two closest clusters are merged '
    + 'again and again until n_clusters remain. The linkage defines what "closest" means. Memory grows with the '
    + 'square of the number of rows.',
  DBSCAN:
    'Density-based: a row with at least min_samples neighbours within the distance eps is a core point, and clusters '
    + 'are connected regions of core points. Rows in sparse regions are labelled noise (-1). Finds the number of '
    + 'clusters and arbitrary shapes, but uses one density level for the whole dataset, so eps must fit the scale of '
    + 'the data.',
  HDBSCAN:
    'Hierarchical DBSCAN: looks at all density levels at once and keeps the clusters that persist longest. Handles '
    + 'clusters of different density, finds the number of clusters, labels noise (-1) and gives each row a '
    + 'membership probability. Usually the best density-based choice; mainly tuned with min_cluster_size.',
  OPTICS:
    'Orders the rows by density reachability and cuts clusters where the reachability changes steeply (xi). Similar '
    + 'to DBSCAN but copes with varying density and needs no fixed eps. Labels noise (-1). Slower than HDBSCAN on '
    + 'large data.',
  BIRCH:
    'Compresses the data in one pass into a tree of small subclusters (clustering feature tree), then groups the '
    + 'subclusters into n_clusters. Fast and memory-light, suitable for large datasets. Works best for compact, '
    + 'round clusters.',
  GaussianMixture:
    'Models the data as a mixture of n_components Gaussian distributions, fitted with expectation-maximization (EM). '
    + 'Each row gets a probability for every component and is assigned to the most likely one. Clusters can be '
    + 'elliptical and of different sizes (see covariance_type). BIC/AIC help to compare runs with different numbers '
    + 'of components.',
  BayesianGaussianMixture:
    'Gaussian mixture with a prior on the component weights, fitted with variational inference. n_components is '
    + 'only an upper limit: components the data do not need get a weight near zero and stay empty, so the number of '
    + 'clusters is inferred from the data. Each row gets a membership probability.',
  BisectingKMeans:
    'Divisive hierarchical K-Means: starts with all rows in one cluster and repeatedly splits one cluster in two '
    + '(2-means) until n_clusters is reached. Fast on large data and less dependent on the start than plain K-Means; '
    + 'clusters are still assumed to be compact.',
};

export const REDUCTION_HELP = {
  PCA:
    'Principal component analysis: a linear rotation of the data onto the directions of largest variance. Fast, '
    + 'deterministic and keeps global distances. The first few components often hold most of the structure; '
    + '"Explore data with these settings" shows the variance explained per component.',
  UMAP:
    'Uniform Manifold Approximation and Projection (non-linear). Builds a neighbour graph in the original space and '
    + 'lays it out in fewer dimensions so that neighbours stay neighbours. Keeps local structure well and much of '
    + 'the global structure; distances between groups that are far apart are not exact. Good input for density-based '
    + 'clustering.',
  TSNE:
    't-distributed stochastic neighbour embedding (non-linear). Concentrates on keeping close neighbours together '
    + 'and produces clearly separated groups in the plot, but the sizes of the groups and the distances between them '
    + 'carry little meaning. Slow on large data and limited to 2 or 3 dimensions; interpret clusters found on t-SNE '
    + 'output with care.',
  None:
    'No reduction: clustering uses all features after preprocessing. The scatter plot then shows a PCA projection, '
    + 'for display only.',
};

const RANDOM_STATE = 'Seed for the random number generator. The same seed gives the same result on the same data. '
  + 'Change it to see whether the result depends on chance (the Stability tab does this automatically).';
const KMEANS_INIT = 'How the starting centroids are chosen. "k-means++" spreads them over the data (more reliable, '
  + 'the usual choice); "random" picks random rows.';
const N_CLUSTERS = 'Number of clusters to form; every row is assigned to one of them. If you are not sure, try a '
  + 'range of values with the Parameter Sweep and compare the metrics.';
const NEIGHBOR_ALGORITHM = 'Data structure for the neighbour search: "auto" chooses one, "ball_tree" and "kd_tree" '
  + 'are search trees (kd_tree is fast for few dimensions), "brute" compares all pairs. Affects speed only, not the '
  + 'result.';
const NEIGHBOR_METRIC = 'Distance used to find neighbours. "euclidean" is the straight-line distance, "manhattan" '
  + '(= l1) sums the absolute differences, "chebyshev" takes the largest single difference, "minkowski" with the '
  + 'default p = 2 equals euclidean.';
const LEAF_SIZE = 'Leaf size of the search tree used for the neighbour search. Affects speed and memory only, not '
  + 'the result.';
const COVARIANCE_TYPE = 'Shape of each component. "full": every cluster has its own ellipse of any orientation '
  + '(most flexible). "tied": all clusters share one ellipse shape. "diag": ellipses along the axes. "spherical": '
  + 'round clusters, each with its own size.';
const MIXTURE_INIT = 'How the components are initialized before fitting: "kmeans" (default, stable), "k-means++", '
  + '"random", or "random_from_data" (random rows as centres).';
const EMBEDDING_METRIC = 'Distance between rows in the input space. "cosine" compares the shape of the profiles and '
  + 'ignores their overall level; "manhattan" sums the absolute differences.';

export const PARAM_HELP = {
  MiniBatchKMeans: { n_clusters: N_CLUSTERS, init: KMEANS_INIT, random_state: RANDOM_STATE },
  KMeans: {
    n_clusters: N_CLUSTERS,
    init: KMEANS_INIT,
    algorithm: '"lloyd" is the standard K-Means iteration. "elkan" skips distance calculations using the triangle '
      + 'inequality; it can be faster with well-separated clusters but uses more memory. Both give the same result.',
    random_state: RANDOM_STATE,
  },
  AffinityPropagation: {
    damping: 'Share of the previous message values kept in each iteration (0.5–1). Higher values make the updates '
      + 'slower but more stable; increase it if the algorithm does not converge.',
    preference: 'How strongly each row wants to be an exemplar. More negative values give fewer clusters; values '
      + 'closer to 0 give more clusters.',
    affinity: 'How the similarity between rows is measured. "euclidean" uses the negative squared Euclidean distance.',
  },
  MeanShift: {
    bandwidth: 'Radius of the window used to estimate the density. Small values find many small clusters, large '
      + 'values merge them. Leave empty (None) to estimate it from the data.',
    cluster_all: 'On: every row is assigned to the nearest cluster, including rows far from any density peak. '
      + 'Off: those rows are labelled noise (-1).',
  },
  SpectralClustering: {
    n_clusters: N_CLUSTERS,
    affinity: 'How the similarity graph is built. "nearest_neighbors" connects each row to its n_neighbors closest '
      + 'rows (sparse, moderate memory). "rbf" uses a Gaussian kernel on all pairs of rows (dense; memory grows with '
      + 'the square of the number of rows).',
    n_neighbors: 'Number of neighbours each row is connected to in the graph (only used with the affinity '
      + '"nearest_neighbors"). Small values follow fine structure; large values give smoother, larger clusters.',
    assign_labels: 'How the final labels are taken from the spectral embedding. "kmeans" runs K-Means on it (common, '
      + 'depends on the seed); "discretize" is less sensitive to the random start.',
    random_state: RANDOM_STATE,
  },
  AgglomerativeClustering: {
    n_clusters: N_CLUSTERS,
    linkage: 'How the distance between two clusters is measured when choosing the next merge. "ward": the merge that '
      + 'increases the within-cluster variance least (compact clusters; euclidean only). "complete": largest distance '
      + 'between members. "average": mean distance between members. "single": smallest distance (can form long chains).',
    metric: 'Distance between rows. Ward linkage needs "euclidean". "cosine" compares the shape of the profiles and '
      + 'ignores their overall level; "manhattan" (= l1) sums the absolute differences; "l2" equals euclidean.',
  },
  DBSCAN: {
    eps: 'Maximum distance between two rows for them to count as neighbours. The most important DBSCAN setting: too '
      + 'small and most rows become noise, too large and clusters merge. It depends on the scale of the data after '
      + 'preprocessing and reduction.',
    min_samples: 'Number of rows (the row itself included) that must lie within eps for a row to be a core point. '
      + 'Higher values require denser clusters and label more rows as noise.',
    metric: NEIGHBOR_METRIC,
    leaf_size: LEAF_SIZE,
    algorithm: NEIGHBOR_ALGORITHM,
  },
  HDBSCAN: {
    min_cluster_size: 'Smallest group of rows that counts as a cluster; smaller groups become noise or join a larger '
      + 'cluster. The main HDBSCAN setting: larger values give fewer, bigger clusters.',
    min_samples: 'How conservative the clustering is. Higher values keep only dense cores and label more rows as '
      + 'noise. Leave empty (None) to use the value of min_cluster_size.',
    cluster_selection_epsilon: 'Clusters closer together than this distance are merged. Helps when HDBSCAN splits a '
      + 'group into many small clusters. 0 turns it off.',
    max_cluster_size: 'Upper limit for the size of a cluster (only with the "eom" selection method). Leave empty '
      + '(None) for no limit.',
    metric: NEIGHBOR_METRIC,
    leaf_size: LEAF_SIZE,
    cluster_selection_method: 'How clusters are taken from the cluster tree. "eom" (excess of mass) keeps the most '
      + 'persistent clusters, often a few large ones. "leaf" takes the leaves of the tree: many small, homogeneous '
      + 'clusters.',
  },
  OPTICS: {
    min_samples: 'Number of rows in a neighbourhood for a row to be a core point. Higher values smooth the '
      + 'reachability and give fewer clusters.',
    xi: 'Minimum steepness in the reachability plot that marks a cluster boundary (0–1). Smaller values find more and '
      + 'smaller clusters; larger values keep only very clear ones.',
    min_cluster_size: 'Smallest cluster as a fraction of all rows (0.05 = 5%).',
    metric: NEIGHBOR_METRIC,
    predecessor_correction: 'Corrects cluster boundaries using the predecessor of each row in the ordering, as in the '
      + 'original OPTICS-ξ method. Usually left on.',
    algorithm: NEIGHBOR_ALGORITHM,
  },
  BIRCH: {
    n_clusters: 'Number of clusters formed from the subclusters in the final step. Leave empty (None) to use the '
      + 'subclusters themselves as the clusters.',
    threshold: 'Maximum radius of a subcluster: a row joins a subcluster only if its radius stays below this value. '
      + 'Smaller values give more, finer subclusters (slower); the right value depends on the scale of the data.',
    branching_factor: 'Maximum number of subclusters in one tree node; a fuller node is split. Mainly affects speed '
      + 'and memory.',
    compute_labels: 'Assign a cluster to every row. The results need the labels, so keep it on.',
  },
  GaussianMixture: {
    n_components: 'Number of Gaussian components, i.e. clusters. Compare runs with different values by BIC/AIC '
      + '(lower is better).',
    covariance_type: COVARIANCE_TYPE,
    init_params: MIXTURE_INIT,
    random_state: RANDOM_STATE,
  },
  BayesianGaussianMixture: {
    n_components: 'Maximum number of components. Choose it generously: components the data do not need stay empty, '
      + 'so the number of clusters found is usually smaller.',
    weight_concentration_prior_type: '"dirichlet_process" (stick-breaking) tends to use few components and leave the '
      + 'others empty. "dirichlet_distribution" spreads the weight more evenly over the components.',
    weight_concentration_prior: 'Strength of the weight prior. Small values let the model leave more components empty '
      + '(fewer clusters); large values favour using more. Empty (None) uses 1 / n_components.',
    covariance_type: COVARIANCE_TYPE,
    init_params: MIXTURE_INIT,
    max_iter: 'Maximum number of iterations. Increase it when a convergence warning appears.',
    random_state: RANDOM_STATE,
  },
  BisectingKMeans: {
    n_clusters: N_CLUSTERS,
    bisecting_strategy: 'Which cluster is split next: "biggest_inertia" splits the cluster with the largest spread '
      + '(sum of squared distances to its centre), "largest_cluster" the one with the most rows.',
    init: 'How the two starting centres of each split are chosen. "k-means++" spreads them out; "random" picks random '
      + 'rows.',
    random_state: RANDOM_STATE,
  },
  PCA: {
    n_components: 'Number of principal components to keep. All of them are used for clustering; the scatter plot '
      + 'shows the first two.',
    svd_solver: 'How the components are computed. "auto" chooses by data size, "full" is exact, "arpack" and '
      + '"randomized" are faster for large data when only a few components are kept. The results are practically '
      + 'the same.',
    random_state: 'Seed; only used by the "arpack" and "randomized" solvers.',
  },
  UMAP: {
    n_components: 'Number of output dimensions. All of them are used for clustering; the plot shows the first two. '
      + 'For clustering, 5–10 dimensions often work better than 2.',
    n_neighbors: 'Size of the neighbourhood UMAP considers for each row. Small values keep fine local structure (many '
      + 'small groups); large values keep more of the global layout.',
    min_dist: 'How closely points may be packed in the output. Small values (0–0.1) give dense clumps, which helps '
      + 'clustering; larger values spread the points more evenly, which can be easier to look at.',
    metric: EMBEDDING_METRIC,
    random_state: RANDOM_STATE,
  },
  TSNE: {
    n_components: 'Number of output dimensions (2 or 3).',
    perplexity: 'Roughly the number of close neighbours each row tries to keep. Usual values are 5–50, and it must be '
      + 'smaller than the number of rows. Larger values show more of the global structure.',
    learning_rate: 'Step size of the optimization. Too small and the points stay in dense clumps; too large and the '
      + 'layout looks like an even ball. 200 is usually fine; larger datasets can need larger values.',
    n_iter: 'Number of optimization steps. Increase it if the layout still looks unsettled.',
    metric: EMBEDDING_METRIC,
    random_state: RANDOM_STATE,
  },
};

const formatValue = (value) => (value === null || value === undefined ? 'None' : String(value));

// Explanation of one parameter plus its default and allowed range from the backend definition.
export function paramHelp(method, definition) {
  const text = PARAM_HELP[method]?.[definition.name] || '';
  const facts = [`Default: ${formatValue(definition.default)}.`];
  if (definition.type === 'select') facts.push(`Options: ${definition.options.map(formatValue).join(', ')}.`);
  else if (definition.min !== undefined && definition.max !== undefined) facts.push(`Allowed: ${definition.min} to ${definition.max}.`);
  if (definition.nullable) facts.push('Can be left empty (None).');
  return [text, facts.join(' ')].filter(Boolean).join('\n');
}

export const HELP = {
  // 1. Upload
  upload: 'One row per item (e.g. protein, gene or sample) and one column per measurement. CSV, TSV, XLSX or XLS, up '
    + 'to 50 MB. The file is kept on the server for this session only and is deleted when you close the page.',
  example: 'Bundled example datasets. Loading one also sets the name column, the data start column and suitable '
    + 'preprocessing settings.',
  sheet: 'Worksheet of the Excel file that holds the data.',
  nameColumn: 'Column with the identifier of each row (e.g. gene or protein name). It labels the points in plots, '
    + 'tables and downloads and is not used for clustering.',
  startIndex: 'Position of the first numeric column, counting from 0. This column and every column after it are used '
    + 'as data. Columns before it (names, annotations, known classes) are ignored by the clustering.',

  // 2. Preprocessing
  preprocessing: 'Prepares the numbers before reduction and clustering. The steps run in the order shown and are '
    + 'saved with every run, in the report and in reproduce.py.',
  maxMissing: 'Rows with a larger share of missing cells are removed before anything else. 100 keeps every row.',
  imputation: 'Value used for the missing cells that remain. "0" treats missing as absent; the column minimum or half '
    + 'of it treats missing as "below the detection limit"; the median is neutral but can blur differences.',
  transform: 'Applied to every value. log(x + 1) compresses large values so that a few very high intensities do not '
    + 'dominate; it needs values above −1. The signed log works with negative values too.',
  scaling: 'Makes values comparable. Per-row scaling compares the shape of each profile and ignores its overall '
    + 'level. Per-column scaling puts every measurement on the same scale, which suits columns of different kinds.',
  removeZeroRows: 'Removes rows in which every value is 0 after imputation; they carry no profile shape and cannot be '
    + 'scaled per row.',
  removeLowVariance: 'Removes columns whose standard deviation (after the transformation) is at or below the '
    + 'threshold. With 0 only constant columns are removed.',
  correlationFilter: 'Removes a column when its correlation |r| with an earlier, kept column is above the threshold, '
    + 'so near-duplicate measurements are counted once.',
  explore: 'Shows the data after these preprocessing settings, without clustering: missing values, PCA variance '
    + 'explained, value distributions before and after scaling, column correlations and outlying rows.',

  // 3./4. Methods
  reduction: 'Optionally maps the data to fewer dimensions before clustering. This can remove noise and speed up '
    + 'clustering. All chosen dimensions are used for clustering; the plots show the first two.',
  clustering: 'Groups the rows. Algorithms differ in what a cluster is: close to a centre (K-Means family), dense '
    + 'regions (DBSCAN, HDBSCAN, OPTICS), merged step by step (hierarchical) or a probability distribution '
    + '(mixture models).',
  useDefaults: 'Uses the default value of every parameter. Untick to change individual parameters; hover over the "?" '
    + 'next to a parameter to see what it does.',
  learnMore: 'Opens a longer guide to all methods and their parameters.',
  check: 'Checks the data and the settings without clustering: missing and non-numeric values, constant columns, '
    + 'duplicates, and whether the chosen methods are suitable for this data size on the server.',
  run: 'Starts the analysis as a background job. The data are checked first; if a setting is unsuitable you are '
    + 'asked before the run starts.',

  // Results
  results: 'Results of the selected run. Use Run History to switch to another run.',
  downloadZip: 'One ZIP with the cluster assignments, cluster summaries, characterization and stability tables, '
    + 'figures (PNG and SVG), all settings, an HTML report and reproduce.py to repeat the run outside the web app.',
  clustersFound: 'Number of clusters, not counting noise (label -1).',
  silhouette: 'How much closer each row is to its own cluster than to the next cluster, averaged over the rows. '
    + 'Range −1 to 1; higher is better. Around 0.5 or more is a clear structure, below 0.25 a weak one. Favours round, '
    + 'well-separated clusters. Noise rows are left out.',
  calinskiHarabasz: 'Ratio of the spread between clusters to the spread within clusters. Higher is better. Has no '
    + 'fixed scale, so only compare runs on the same data.',
  daviesBouldin: 'Average similarity of each cluster to its most similar other cluster (size relative to distance). '
    + '0 is the best value; lower is better.',
  dbcv: 'Density-based clustering validation. Rates how dense the clusters are compared with the gaps between them, '
    + 'so it also works for non-round clusters. Range −1 to 1; higher is better. Most useful for DBSCAN, HDBSCAN and '
    + 'OPTICS.',
  bicAic: 'Information criteria of mixture models: how well the model fits, with a penalty for the number of '
    + 'parameters. Lower is better. Only compare runs on the same data, e.g. to choose n_components.',
  meanMembership: 'Average probability with which the clustered rows belong to their cluster. Close to 1 means '
    + 'clear assignments; lower values mean many rows lie between clusters.',
  componentsUsed: 'Number of mixture components that received rows. The rest were not needed by the data.',
  runtime: 'Time for preprocessing, reduction and clustering of this run.',

  // Tabs
  tabScatter: 'Every row as a point, colored by cluster, in the first two dimensions of the reduction (or a PCA '
    + 'projection without reduction). Hover over a point for its name; drag to zoom.',
  tabNetwork: 'One node per cluster, placed at the cluster centre; edges connect clusters, thicker for closer '
    + 'clusters. Click a cluster to open its Sub-network View.',
  tabSubnetwork: 'Network of the rows of one cluster, each connected to its nearest neighbours. Draw a lasso around '
    + 'points to plot their raw profiles. Becomes available after clicking a cluster in the Clusters Network.',
  tabHeatmap: 'Preprocessed values as colors: a summary row per cluster (median or mean), or the rows of one cluster. '
    + 'The optional dendrogram only orders the rows; it is not another clustering.',
  tabCharacterization: 'What makes each cluster special: size, typical members, profile with spread, the features '
    + 'that distinguish it from the other rows, and outliers. Computed on all rows.',
  tabStability: 'Repeats the clustering on random subsamples and with other seeds and measures how much of each '
    + 'cluster survives. Shows whether a cluster is a real pattern or depends on chance.',
  methodDetails: 'Summary of the settings of this run, with a methods paragraph you can adapt for a paper.',
  clusterDetails: 'One table row per data row with its cluster, plot coordinates and color. Click a column header to '
    + 'sort. Membership and stability columns appear when they are available.',

  // Run management
  runHistory: 'Every run of this session, newest first. View shows a run\'s results and puts its settings back into '
    + 'the form; for failed runs, Load settings does the same. Details lists all settings. Labels TSV and ZIP '
    + 'download the results. Runs are deleted when you close the page.',
  leaderboard: 'Ranks the runs by a quality measure, after removing runs that break the constraints you set. The best '
    + 'run depends on the chosen measure; no single score finds the best clustering in general.',
  rankBy: 'Silhouette, Calinski–Harabasz and DBCV: higher is better. Davies–Bouldin and runtime: lower is better. '
    + 'Stability: agreement of repeated solutions (needs a stability analysis). Pareto: several measures at once.',
  pareto: 'A run is on the first front (F1) when no other run is at least as good on every chosen measure and better '
    + 'on one. F2 is the front after removing F1, and so on.',
  constraints: 'Runs that break a limit are excluded from the ranking (tick "Show excluded runs" to see why). Leave a '
    + 'field empty for no limit.',
  maxNoise: 'Maximum share of rows labelled noise (-1).',
  minClusterSize: 'Smallest allowed cluster (number of rows).',
  maxImbalance: 'Largest cluster size divided by the smallest cluster size.',
  minStability: 'Minimum mean ARI from the stability analysis; runs without a stability analysis are excluded.',
  recommended: 'A run gets the "Recommended" badge only when both its quality (Silhouette or DBCV) and its stability '
    + '(mean ARI) reach the thresholds. Runs without a stability analysis cannot get it.',
  stabilityScore: 'Mean adjusted Rand index (ARI) between the run and its repeated solutions. 1 means the clusters '
    + 'always come back the same; values below about 0.75 mean they change noticeably.',
  pin: 'Keeps the current best run in view while you try other settings.',

  // Comparison
  compare: 'Puts two runs side by side and measures how much their clusterings agree, on all rows.',
  alignColors: 'Gives every Run B cluster the color of the Run A cluster it overlaps most, so matching clusters look '
    + 'the same.',
  ari: 'Adjusted Rand index: agreement of two clusterings over all pairs of rows, corrected for chance. 1 = '
    + 'identical, 0 = no more agreement than random.',
  nmi: 'Normalized mutual information: how much knowing the cluster in one run tells you about the cluster in the '
    + 'other. 0 = nothing, 1 = identical.',
  changedSamples: 'Rows that are not in the Run B cluster matched to their Run A cluster. Cluster numbers themselves '
    + 'have no meaning across runs, so clusters are matched by overlap first.',
  flow: 'Contingency matrix: how many rows of each Run A cluster end up in each Run B cluster. The Sankey diagram '
    + 'shows the same numbers as flows.',
  matching: 'Each Run A cluster is paired with the Run B cluster it shares the most rows with (one-to-one, by '
    + 'Jaccard overlap = shared rows / rows in either cluster).',

  // Stability tab
  resamples: 'Number of random subsamples (80% of the rows each) that are clustered again with the same settings.',
  seeds: 'Number of extra runs on all rows with other random seeds. Only for algorithms that use a seed.',
  jaccard: 'For each cluster: overlap with its best-matching cluster in every repeated solution (shared rows / rows '
    + 'in either), averaged. ≥ 0.75 stable, 0.6–0.75 doubtful, below 0.6 unstable.',
  agreement: 'ARI and NMI of the repeated solutions against this run and against each other. A wide spread means the '
    + 'result depends on the exact rows or the seed.',
  rowStability: 'Share of the repeated solutions in which a row stays in its (matched) cluster. Noise rows count as '
    + 'stable when they stay noise.',
  consensus: 'The cluster a row was assigned to most often across the repeated solutions; "Agreement" is how often.',

  // Sweep
  sweep: 'Runs the selected algorithm several times with different values of one or two parameters, keeping all '
    + 'other settings fixed, so the metrics of the trials can be compared. Every trial becomes a normal run.',
  sweepRange: 'Values from "from" to "to" in steps of "step" (both ends included).',
  sweepNone: 'Also try the parameter left empty (None), which usually means "choose automatically".',
  maxTrials: 'Upper limit of trials (at most 30). If there are more combinations, a random subset is tried.',
  screening: 'Runs every trial on a subsample first, then repeats only the best ones on all rows. Saves time for '
    + 'large datasets.',
  screeningFraction: 'Share of the rows used for the screening runs.',
  screeningTopK: 'Number of best screening trials that are repeated on all rows and saved as runs.',
  screeningMetric: 'Measure used to pick the best screening trials.',

  // Characterization
  medoid: 'The member with the smallest total distance to the other members: the most typical row of the cluster.',
  representatives: 'The members closest to the medoid.',
  profiles: 'Median value per column of each selected cluster, with the band between the 25th and 75th percentile.',
  cliffsDelta: 'Cliff\'s δ compares the values of one cluster with all other rows: +1 means every value in the cluster '
    + 'is higher, −1 every value lower, 0 no difference. |δ| above about 0.47 is usually called a large effect.',
  kruskal: 'Kruskal–Wallis test per column across the clusters, with ε² as effect size (0 to 1) and Benjamini–'
    + 'Hochberg adjusted q-values. Descriptive only: the clusters were built to differ, so the p-values are not '
    + 'proof of a difference.',
  outliers: 'Rows furthest from their cluster\'s median profile, as a robust z-score (distance compared with the '
    + 'typical distance in the cluster). Values above about 3.5 are unusual.',
  members: 'Copy or download the names of a cluster, one per line, e.g. for g:Profiler or STRING.',
  boxplot: 'Distribution of one column in every cluster (box: middle 50% of the rows, line: median).',

  // Preflight
  preflight: 'Checks of the data and of the settings for this data size. "Recommended", "May be slow" and "Not '
    + 'recommended" refer to the hosted server and its time and memory limits.',
};

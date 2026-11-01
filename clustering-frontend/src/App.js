import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import Modal from 'react-modal';
// import Plot from 'react-plotly.js-basic-dist';
import Plot from 'react-plotly.js';
import { readString } from 'react-papaparse';
import * as XLSX from 'xlsx';
import './App.css';
import AlgorithmGuideDR from './AlgorithmGuideDR';
import AlgorithmGuideClustering from './AlgorithmGuideClustering';
import RunHistory from './RunHistory';
import RunComparison from './RunComparison';
import Leaderboard from './Leaderboard';
import PreflightReport from './PreflightReport';
import { ApiError, fetchJson, formatApiError, pollJob } from './api';
import PreprocessingOptions, { DEFAULT_PREPROCESSING } from './PreprocessingOptions';
import ExploratoryAnalysis from './ExploratoryAnalysis';
import ClusterCharacterization from './ClusterCharacterization';
import StabilityAnalysis from './StabilityAnalysis';
import ParameterSweep from './ParameterSweep';
import HelpTip from './HelpTip';
import { ALGORITHM_HELP, HELP, REDUCTION_HELP, paramHelp } from './helpTexts';

// Same-origin in both modes: nginx proxies this path in Docker, src/setupProxy.js in the dev server.
const API_BASE = process.env.REACT_APP_API_BASE || '/rapcluster/api';

const createSessionId = () => (
  window.crypto?.randomUUID
    ? window.crypto.randomUUID().replace(/-/g, '')
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 12)}`
);


const methodDescriptions = {
  clustering: {
    MiniBatchKMeans: "Fast, memory-efficient K-Means variant for large datasets.",
    KMeans: "Partitions data into k clusters minimizing within-cluster variance.",
    AffinityPropagation: "Message-passing clustering that identifies exemplars.",
    MeanShift: "Clusters based on data density without predefining cluster number.",
    SpectralClustering: "Graph-based clustering using Laplacian eigenmaps.",
    AgglomerativeClustering: "Hierarchical clustering using bottom-up merging.",
    DBSCAN: "Density-based clustering that labels noise and finds dense areas.",
    HDBSCAN: "Improved DBSCAN for varying density with hierarchical clustering.",
    OPTICS: "Orders data to identify clusters at varying density thresholds.",
    BIRCH: "Scalable clustering using compact feature trees.",
    GaussianMixture: "Probabilistic clustering assuming Gaussian-distributed data.",
    BayesianGaussianMixture: "Gaussian mixture that infers how many of the allowed components are needed; soft assignments.",
    BisectingKMeans: "Divisive hierarchical K-Means: repeatedly splits one cluster in two; scales to large data."
  },
  reduction: {
    PCA: "Linear projection preserving maximal variance in fewer dimensions.",
    UMAP: "Non-linear projection preserving local and global data structure.",
    TSNE: "Projects high-dimensional data by preserving local similarities.",
    None: "No dimensionality reduction; clustering uses all features."
  }
};

function App() {
  const [file, setFile] = useState(null);
  const [methodDetails, setMethodDetails] = useState('');
  const [networkPlotData, setNetworkPlotData] = useState([]);
  const [networkPlotLayout, setNetworkPlotLayout] = useState({});
  const [subnetworkData, setSubnetworkData] = useState([]);
  const [topNNeighbors, setTopNNeighbors] = useState(5);
  const [subnetworkLayout, setSubnetworkLayout] = useState({
    autosize: true,
    hovermode: 'closest',
    dragmode: 'lasso', // Default drag mode
    title: 'Select a Cluster to View Sub-network',
    xaxis: { title: 'Reduced Dimension 1' },
    yaxis: { title: 'Reduced Dimension 2' },
    margin: { l: 40, r: 40, b: 80, t: 80 },
    plot_bgcolor: '#fcfcfc',
    paper_bgcolor: '#fcfcfc',
    height: 600,
    showlegend: true
});
  const [selectedSubnetwork, setSelectedSubnetwork] = useState(null);
  const [showReductionModal, setShowReductionModal] = useState(false);
  const [showClusteringModal, setShowClusteringModal] = useState(false);
  const [selectedProfilesScatterData, setSelectedProfilesScatterData] = useState([]);

  const closeModals = () => {
    setShowReductionModal(false);
    setShowClusteringModal(false);
  };
  const [activeTab, setActiveTab] = useState("scatter");
  const [fileName, setFileName] = useState('');
  const [fileType, setFileType] = useState('');
  const [columnNames, setColumnNames] = useState([]);
  const [excelSheetNames, setExcelSheetNames] = useState([]);
  const [excelColumnsBySheet, setExcelColumnsBySheet] = useState({});
  const [selectedSheetName, setSelectedSheetName] = useState('');
  const [nameColumn, setNameColumn] = useState('Name');
  const [intensityStartIndex, setIntensityStartIndex] = useState(1); 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [displayInfo, setDisplayInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [clusterColors, setClusterColors] = useState({});
  const [intensityColumns, setIntensityColumns] = useState([]);
  const [showClusterDetails, setShowClusterDetails] = useState(true);
  const [detailsSort, setDetailsSort] = useState({ key: 'cluster', direction: 'asc' });
  const [heatmapCluster, setHeatmapCluster] = useState('summary');
  const [heatmapAggregation, setHeatmapAggregation] = useState('median');
  const [heatmapRowLimit, setHeatmapRowLimit] = useState(250);
  const [showHeatmapDendrogram, setShowHeatmapDendrogram] = useState(true);

  const [availableAlgos, setAvailableAlgos] = useState([]);
  const [availableReductions, setAvailableReductions] = useState([]);
  const [algoParams, setAlgoParams] = useState({}); 
  const [reductionParamsDef, setReductionParamsDef] = useState({}); 

  const [selectedReduction, setSelectedReduction] = useState('None');
  const [customReductionParams, setCustomReductionParams] = useState({});
  const [useDefaultReductionParams, setUseDefaultReductionParams] = useState(true);

  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [customAlgorithmParams, setCustomAlgorithmParams] = useState({});
  const [useDefaultAlgorithmParams, setUseDefaultAlgorithmParams] = useState(true);

  const [selectedProfilesData, setSelectedProfilesData] = useState([]);
  const [selectedProfilesLayout, setSelectedProfilesLayout] = useState({});
  const [selectedProfileNames, setSelectedProfileNames] = useState([]);

  // Run history: the dataset is uploaded once per session and every run is stored server-side.
  const [sessionId] = useState(createSessionId);
  const datasetIdRef = useRef(null);
  const [runs, setRuns] = useState([]);
  const [activeRunId, setActiveRunId] = useState(null);
  const [preflightReport, setPreflightReport] = useState(null);
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
  const [examples, setExamples] = useState([]);
  const [selectedExampleId, setSelectedExampleId] = useState('');
  // Server limits; the defaults apply until /api/limits answers.
  const [limits, setLimits] = useState({ max_upload_mb: 50 });
  const [jobProgress, setJobProgress] = useState(null);
  const [sampling, setSampling] = useState(null);
  const [clusterProfiles, setClusterProfiles] = useState(null);
  // Preprocessing and the values that were clustered, e.g. "log2(x + 1), per-row z-score".
  const [preprocessingOptions, setPreprocessingOptions] = useState({ ...DEFAULT_PREPROCESSING });
  const [exploreData, setExploreData] = useState(null);
  const [valueLabel, setValueLabel] = useState('log2(x + 1), per-row z-score');
  const activeRun = runs.find(run => run.run_id === activeRunId);

  const fileInputRef = useRef(null);

  const parseDelimitedHeaderLine = (line, delimiter) => {
    const columns = [];
    let currentValue = '';
    let inQuotes = false;

    for (let index = 0; index < line.length; index += 1) {
      const character = line[index];

      if (character === '"') {
        if (inQuotes && line[index + 1] === '"') {
          currentValue += '"';
          index += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (character === delimiter && !inQuotes) {
        columns.push(currentValue.trim());
        currentValue = '';
      } else {
        currentValue += character;
      }
    }

    columns.push(currentValue.trim());

    return columns
      .map(column => column.replace(/^"(.*)"$/, '$1').trim())
      .filter(Boolean);
  };


  async function readFirstLine(file) {
    const reader = file.stream().getReader();
    const decoder = new TextDecoder();

    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        // File ended before newline
        return buffer || null;
      }

      buffer += decoder.decode(value, { stream: true });

      const newlineIndex = buffer.search(/\r?\n/);

      if (newlineIndex >= 0) {
        await reader.cancel();

        return buffer
          .slice(0, newlineIndex)
          .replace(/^\uFEFF/, '');
      }
    }
  }

  const getDelimitedColumnNames = async (selectedFile, ext) => {
    const header = await readFirstLine(selectedFile);

    const delimiter = ext === 'tsv' ? '\t' : ',';
    return parseDelimitedHeaderLine(header, delimiter);
  };

  const getExcelMetadata = async (selectedFile) => {
    const fileBuffer = await selectedFile.arrayBuffer();
    const workbook = XLSX.read(fileBuffer, { type: 'array' });
    const sheetNames = workbook.SheetNames || [];
    const columnsBySheet = {};

    sheetNames.forEach(sheetName => {
      const worksheet = workbook.Sheets[sheetName];
      const rows = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: '', blankrows: false });
      const headerRow = rows.find(row => Array.isArray(row) && row.some(cell => String(cell).trim().length > 0)) || [];
      columnsBySheet[sheetName] = headerRow
        .map(cell => String(cell).trim())
        .filter(Boolean);
    });

    return { sheetNames, columnsBySheet };
  };

  const handleNameColumnChange = (newNameColumn) => {
    setNameColumn(newNameColumn);
  };

  const handleExcelSheetChange = (sheetName) => {
    setSelectedSheetName(sheetName);
    const headers = excelColumnsBySheet[sheetName] || [];
    setColumnNames(headers);
    setNameColumn(previousNameColumn => {
      const updatedNameColumn = headers.includes(previousNameColumn)
        ? previousNameColumn
        : (headers[0] || 'Name');
      return updatedNameColumn;
    });
  };

  useEffect(() => {
    const fetchAlgorithms = async () => {
      try {
        const response = await fetch(`${API_BASE}/algorithms`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAvailableAlgos(data.clustering_algorithms);
        setAvailableReductions(['None', ...data.dimensionality_reduction_methods]);
        setAlgoParams(data.algorithm_params);
        setReductionParamsDef(data.reduction_params);
        
        // Set initial defaults if available
        if (data.clustering_algorithms.length > 0) {
          setSelectedAlgorithm(data.clustering_algorithms[0]);
        }
      } catch (e) {
        console.error("Failed to fetch algorithms:", e);
        setError("Failed to load clustering algorithms. Is the backend running?");
      }
    };
    fetchAlgorithms();
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/examples`)
      .then(response => (response.ok ? response.json() : { examples: [] }))
      .then(data => setExamples(data.examples || []))
      .catch(e => console.error("Failed to fetch example datasets:", e));
  }, []);

  useEffect(() => {
    fetchJson(`${API_BASE}/limits`)
      .then(setLimits)
      .catch(e => console.error("Failed to fetch server limits:", e));
  }, []);

  // A pending "Run anyway" confirmation only applies to the settings it was checked with.
  useEffect(() => {
    setAwaitingConfirmation(false);
  }, [
    nameColumn, intensityStartIndex, selectedSheetName, selectedReduction, customReductionParams,
    useDefaultReductionParams, selectedAlgorithm, customAlgorithmParams, useDefaultAlgorithmParams, preprocessingOptions,
  ]);

  // Closing (or reloading) the page deletes the session's datasets and runs on the server.
  useEffect(() => {
    const closeSession = () => {
      navigator.sendBeacon?.(`${API_BASE}/sessions/${sessionId}/close`);
    };
    window.addEventListener('pagehide', closeSession);
    return () => window.removeEventListener('pagehide', closeSession);
  }, [sessionId]);

  useEffect(() => {
    if (
      activeTab === "subnetwork" &&
      selectedSubnetwork !== null &&
      subnetworkData.length > 0
    ) {
      handleSubnetworkClick(selectedSubnetwork, clusterColors);
    }
  }, [topNNeighbors]);

  const handleDownloadTSV = () => {
    if (!clusterData || clusterData.length === 0) {
      alert("No cluster data to download."); 
      return;
    }
    // The table only holds a sample of large datasets; the server file has every row.
    if (sampling?.sampled && activeRun) {
      const link = document.createElement('a');
      link.href = `${API_BASE}${activeRun.links.labels_tsv}`;
      link.setAttribute('download', '');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return;
    }

    const headers = ["Name", "Cluster", "X", "Y", "Color",
      ...(hasMembership ? ["Membership"] : []), ...(hasRowStability ? ["RowStability"] : [])];
    const tsvRows = sortedClusterData.map(d => [
      d.name,
      d.cluster === -1 ? 'Noise (-1)' : d.cluster, 
      d.x.toFixed(6), 
      d.y.toFixed(6),
      d.color,
      ...(hasMembership ? [d.membership ?? ''] : []), ...(hasRowStability ? [d.stability ?? ''] : []),
    ]);

    const allRows = [headers.join('\t'), ...tsvRows.map(row => row.join('\t'))];
    const tsvContent = allRows.join('\n');

    const blob = new Blob([tsvContent], { type: 'text/tab-separated-values;charset=utf-8;' });

    const link = document.createElement('a');
    if (link.download !== undefined) { 
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', 'cluster_details.tsv');
      link.style.visibility = 'hidden'; 
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url); 
    } else {
      alert("Your browser does not support automatic downloads. Please copy the data manually.");
      console.error("Browser does not support download attribute.");
    }
  };
  const handleFileChange = (e) => loadSelectedFile(e.target.files[0]);

  const loadSelectedFile = async (selectedFile) => {
    datasetIdRef.current = null;
    setPreflightReport(null);
    setExploreData(null);
    setAwaitingConfirmation(false);
    if (selectedFile && selectedFile.size > limits.max_upload_mb * 1024 * 1024) {
      setError(
        `${selectedFile.name} is ${(selectedFile.size / 1024 ** 2).toFixed(1)} MB, but the RapCluster server accepts files `
        + `up to ${limits.max_upload_mb} MB (upload limit set by the RapCluster server). `
        + 'Please remove unused columns or rows, or save the table as xlsx, which is usually smaller.'
      );
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      selectedFile = null;
    }
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      const ext = selectedFile.name.split('.').pop().toLowerCase();
      if (['csv', 'tsv', 'xlsx', 'xls'].includes(ext)) {
        const normalizedType = ext === 'xls' ? 'xlsx' : ext;
        setFileType(normalizedType);
        setError(null);

        if (normalizedType === 'csv' || normalizedType === 'tsv') {
          try {
            const headers = await getDelimitedColumnNames(selectedFile, normalizedType);
            setColumnNames(headers);
            setExcelSheetNames([]);
            setExcelColumnsBySheet({});
            setSelectedSheetName('');
            setNameColumn(previousNameColumn => {
              if (headers.includes(previousNameColumn)) {
                return previousNameColumn;
              }
              return headers[0] || previousNameColumn || 'Name';
            });
          } catch (readError) {
            console.error('Failed to read column names from file:', readError);
            setColumnNames([]);
          }
        } else if (normalizedType === 'xlsx') {
          try {
            const { sheetNames, columnsBySheet } = await getExcelMetadata(selectedFile);
            setExcelSheetNames(sheetNames);
            setExcelColumnsBySheet(columnsBySheet);

            const defaultSheet = sheetNames[0] || '';
            const headers = defaultSheet ? (columnsBySheet[defaultSheet] || []) : [];
            const defaultNameColumn = headers[0] || 'Name';

            setSelectedSheetName(defaultSheet);
            setColumnNames(headers);
            setNameColumn(defaultNameColumn);
          } catch (readError) {
            console.error('Failed to read Excel metadata:', readError);
            setExcelSheetNames([]);
            setExcelColumnsBySheet({});
            setSelectedSheetName('');
            setColumnNames([]);
            setError('Could not read workbook sheets/columns from the selected Excel file.');
          }
        } else {
          setColumnNames([]);
          setExcelSheetNames([]);
          setExcelColumnsBySheet({});
          setSelectedSheetName('');
        }
      } else {
        setError('Unsupported file type. Please select a CSV, TSV, or Excel file.');
        setFile(null);
        setFileName('');
        setFileType('');
        setColumnNames([]);
        setExcelSheetNames([]);
        setExcelColumnsBySheet({});
        setSelectedSheetName('');
      }
    } else {
      setFile(null);
      setFileName('');
      setFileType('');
      setColumnNames([]);
      setExcelSheetNames([]);
      setExcelColumnsBySheet({});
      setSelectedSheetName('');
    }
  };

  const handleReductionParamChange = (name, value, type) => {
    setCustomReductionParams(prev => ({
      ...prev,
      [name]: type === 'int' ? parseInt(value) : (type === 'float' ? parseFloat(value) : value)
    }));
    setUseDefaultReductionParams(false);
  };

  const handleAlgorithmParamChange = (name, value, type) => {
    setCustomAlgorithmParams(prev => ({
      ...prev,
      [name]: type === 'int' ? parseInt(value) : (type === 'float' ? parseFloat(value) : value)
    }));
    setUseDefaultAlgorithmParams(false);
  };

  const renderParams = (paramsDef, currentParams, setter, useDefault, selectedAlgoName) => {
    if (!paramsDef || !paramsDef[selectedAlgoName]) return null;

    const definitions = paramsDef[selectedAlgoName];
    const defaultVals = {};
    definitions.forEach(p => {
      defaultVals[p.name] = p.default;
    });

    return (
      <div className="param-grid">
        {definitions.map(param => (
          <div className="param-item" key={param.name}>
            <label htmlFor={`${selectedAlgoName}-${param.name}`}>
              {param.name}:<HelpTip text={paramHelp(selectedAlgoName, param)} label={param.name} />
            </label>
            {param.type === 'int' && (
              <input
                type="number"
                id={`${selectedAlgoName}-${param.name}`}
                value={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.value, 'int')}
                min={param.min}
                max={param.max}
                step={param.step}
              />
            )}
            {param.type === 'float' && (
              <input
                type="number"
                id={`${selectedAlgoName}-${param.name}`}
                value={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.value, 'float')}
                min={param.min}
                max={param.max}
                step={param.step}
              />
            )}
            {param.type === 'boolean' && (
              <input
                type="checkbox"
                id={`${selectedAlgoName}-${param.name}`}
                checked={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.checked, 'boolean')}
              />
            )}
            {param.type === 'select' && (
              <select
                id={`${selectedAlgoName}-${param.name}`}
                value={useDefault ? defaultVals[param.name] : (currentParams[param.name] ?? defaultVals[param.name])}
                onChange={(e) => setter(param.name, e.target.value, 'string')}
              >
                {param.options.map(option => (
                  <option key={option} value={option}>{String(option)}</option>
                ))}
              </select>
            )}
            {param.nullable && (
                <div className="nullable-info">
                  (Leave blank for default `None`)
                </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const formatParamValue = (value) => (value === null || value === undefined ? 'None' : String(value));

  const buildMethodDetails = (algorithm, algorithmValues, reduction, reductionValues) => {
    const lineFor = (values) => (p) => (
      `- ${p.name}: ${formatParamValue(p.name in values ? values[p.name] : p.default)}`
    );

    let details = `Clustering Algorithm: ${algorithm}\n`;
    details += `  → ${methodDescriptions.clustering[algorithm] || 'No description.'}\n`;
    details += `${(algoParams[algorithm] || []).map(lineFor(algorithmValues)).join('\n')}\n\n`;

    if (reduction === 'None') {
      details += `Dimensionality Reduction: None\n`;
      details += `  → ${methodDescriptions.reduction.None}\n`;
    } else {
      details += `Dimensionality Reduction: ${reduction}\n`;
      details += `  → ${methodDescriptions.reduction[reduction] || 'No description.'}\n`;
      details += (reductionParamsDef[reduction] || []).map(lineFor(reductionValues)).join('\n');
    }
    return details;
  };

  const uploadDataset = async () => {
    const formData = new FormData();
    formData.append('file', file);
    const data = await fetchJson(
      `${API_BASE}/sessions/${sessionId}/datasets`,
      { method: 'POST', body: formData },
      { uploading: true, uploadLimitMb: limits.max_upload_mb },
    );
    datasetIdRef.current = data.dataset_id;
    return data.dataset_id;
  };

  const applyRunResult = (data) => {
    setClusterData(data.cluster_data);
    setMetrics(data.metrics);
    setClusterColors(data.cluster_colors);
    setIntensityColumns(data.intensity_columns || []);
    setSampling(data.sampling || null);
    setDisplayInfo(data.display || null);
    setClusterProfiles(data.cluster_profiles || null);
    setValueLabel(data.value_label || 'log2(x + 1), per-row z-score');
    setHeatmapCluster('summary');
    setSubnetworkData([]);
    setSelectedSubnetwork(null);
    setSelectedProfilesData([]);
    setSelectedProfileNames([]);
    setActiveTab(previousTab => (previousTab === 'subnetwork' ? 'scatter' : previousTab));
    setActiveRunId(data.run ? data.run.run_id : null);

    const { data: netData, layout: netLayout } = computeCentroidsAndEdges(data.cluster_data, data.cluster_colors, data.cluster_profiles);
    setNetworkPlotData(netData);
    setNetworkPlotLayout(netLayout);
  };

  // Puts a run's algorithm and reduction settings back into the form so it can be tweaked and re-run.
  const restoreRunSettings = (run) => {
    const matchesDefaults = (definitions, values) => definitions.every(p => values[p.name] === p.default);

    const algorithmDefaults = matchesDefaults(algoParams[run.clustering.algorithm] || [], run.clustering.params);
    setSelectedAlgorithm(run.clustering.algorithm);
    setUseDefaultAlgorithmParams(algorithmDefaults);
    setCustomAlgorithmParams(algorithmDefaults ? {} : run.clustering.params);

    const reductionDefaults = matchesDefaults(reductionParamsDef[run.reduction.method] || [], run.reduction.params);
    setSelectedReduction(run.reduction.method);
    setUseDefaultReductionParams(reductionDefaults);
    setPreprocessingOptions({ ...DEFAULT_PREPROCESSING, ...(run.preprocessing.options || {}) });
    setCustomReductionParams(reductionDefaults ? {} : run.reduction.params);

    setMethodDetails(buildMethodDetails(
      run.clustering.algorithm, run.clustering.params, run.reduction.method, run.reduction.params
    ));
  };

  const handleViewRun = async (run) => {
    setError(null);
    setLoading(true);
    try {
      const data = await fetchJson(`${API_BASE}${run.links.result}`);
      applyRunResult(data);
      restoreRunSettings(run);
    } catch (e) {
      console.error("Loading run failed:", e);
      setError(`Could not load run #${run.number}: ${formatApiError(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const collectRunSettings = () => {
    // Determine which parameters to send for reduction
    let finalReductionParams = {};
    if (!useDefaultReductionParams && selectedReduction !== 'None') {
        const defaultDef = reductionParamsDef[selectedReduction] || [];
        defaultDef.forEach(paramDef => {
            const val = customReductionParams[paramDef.name];
            // Handle nullable 'None' values
            if (paramDef.nullable && (val === '' || val === undefined || val === null)) {
                finalReductionParams[paramDef.name] = 'None';
            } else if (val !== undefined) {
                finalReductionParams[paramDef.name] = val;
            } else {
                finalReductionParams[paramDef.name] = paramDef.default;
            }
        });
    } else if (selectedReduction !== 'None') {
        const defaultDef = reductionParamsDef[selectedReduction] || [];
        defaultDef.forEach(paramDef => {
            finalReductionParams[paramDef.name] = paramDef.default;
        });
    }

    let finalAlgorithmParams = {};
    if (!useDefaultAlgorithmParams && selectedAlgorithm) {
        const defaultDef = algoParams[selectedAlgorithm] || [];
        defaultDef.forEach(paramDef => {
            const val = customAlgorithmParams[paramDef.name];
            if (paramDef.nullable && (val === '' || val === undefined || val === null)) {
                finalAlgorithmParams[paramDef.name] = 'None';
            } else if (val !== undefined) {
                finalAlgorithmParams[paramDef.name] = val;
            } else {
                finalAlgorithmParams[paramDef.name] = paramDef.default;
            }
        });
    } else if (selectedAlgorithm) {
        const defaultDef = algoParams[selectedAlgorithm] || [];
        defaultDef.forEach(paramDef => {
            finalAlgorithmParams[paramDef.name] = paramDef.default;
        });
    }

    return { finalReductionParams, finalAlgorithmParams };
  };

  const buildSettingsFormData = (currentDatasetId, { finalReductionParams, finalAlgorithmParams }, extraFields = {}) => {
    const formData = new FormData();
    formData.append('sessionId', sessionId);
    formData.append('datasetId', currentDatasetId);
    formData.append('nameColumn', nameColumn);
    formData.append('intensityStartIndex', intensityStartIndex);
    formData.append('sheetName', fileType === 'xlsx' ? selectedSheetName : '');
    formData.append('preprocessing', JSON.stringify(preprocessingOptions));
    formData.append('reductionMethod', selectedReduction);
    formData.append('reductionParams', JSON.stringify(finalReductionParams));
    formData.append('clusteringAlgorithm', selectedAlgorithm);
    formData.append('clusteringParams', JSON.stringify(finalAlgorithmParams));
    Object.entries(extraFields).forEach(([key, value]) => formData.append(key, value));
    return formData;
  };

  // Posts the settings for the session's dataset. The file is only uploaded the first time;
  // if the server-side copy expired it is uploaded once more and the request is retried.
  const postWithDataset = async (path, settings, extraFields) => {
    const send = (currentDatasetId) => fetchJson(`${API_BASE}${path}`, {
      method: 'POST',
      body: buildSettingsFormData(currentDatasetId, settings, extraFields),
    });

    try {
      return await send(datasetIdRef.current || await uploadDataset());
    } catch (e) {
      if (e instanceof ApiError && e.code === 'dataset_not_found') {
        return send(await uploadDataset());
      }
      throw e;
    }
  };

  const validateInputs = () => {
    if (!file) {
      setError('Please select a file first.');
      return false;
    }
    if (fileType === 'xlsx' && !selectedSheetName) {
      setError('Please select an Excel sheet first.');
      return false;
    }
    return true;
  };

  const runPreflight = async (settings) => {
    const data = await postWithDataset(`/sessions/${sessionId}/preflight`, settings);
    setPreflightReport(data);
    return data;
  };

  const executeRun = async (settings) => {
    setClusterData(null);
    setMetrics(null);
    setClusterColors({});
    setIntensityColumns([]);
    setSampling(null);
    setDisplayInfo(null);
    setClusterProfiles(null);
    setHeatmapCluster('summary');
    setActiveRunId(null);
    setMethodDetails(buildMethodDetails(
      selectedAlgorithm, settings.finalAlgorithmParams, selectedReduction, settings.finalReductionParams
    ));

    console.log("Sending reduction params:", settings.finalReductionParams);
    console.log("Sending clustering params:", settings.finalAlgorithmParams);

    // The server starts a background job (HTTP 202); its status is polled until it finishes.
    const { run } = await postWithDataset('/cluster', settings);
    setRuns(previousRuns => [...previousRuns, run]);
    setJobProgress({ run, stage: 'starting', message: 'Starting the analysis', elapsed: 0 });
    let status;
    try {
      status = await pollJob(API_BASE, run.links.status, {
        onProgress: (update) => setJobProgress({
          run, stage: update.stage, message: update.message, elapsed: update.elapsed_seconds,
        }),
      });
    } finally {
      setJobProgress(null);
    }
    if (status.run) handleRunUpdated(status.run);
    if (status.state !== 'success') {
      const outcome = status.state === 'cancelled' ? 'was cancelled' : 'failed';
      throw new ApiError(`Run #${run.number} ${outcome}: ${status.error}`, { code: status.error_code, hint: status.hint });
    }
    applyRunResult(await fetchJson(`${API_BASE}${run.links.result}`));
  };

  const handleCancelRun = async () => {
    if (!jobProgress) return;
    try {
      await fetchJson(`${API_BASE}${jobProgress.run.links.cancel}?kind=cluster`, { method: 'POST' });
    } catch (e) {
      setError(`Could not cancel run #${jobProgress.run.number}: ${formatApiError(e)}`);
    }
  };

  const formatDuration = (seconds) => (
    seconds >= 60 ? `${Math.round(seconds / 6) / 10} min` : `${Math.round(seconds)} s`
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setAwaitingConfirmation(false);
    if (!validateInputs()) return;

    setLoading(true);
    try {
      const settings = collectRunSettings();
      // Unsuitable algorithm–dataset combinations or data problems need an explicit confirmation.
      const report = await runPreflight(settings);
      if (report.has_data_errors || report.compatibility.overall !== 'recommended') {
        setAwaitingConfirmation(true);
        return;
      }
      await executeRun(settings);
    } catch (e) {
      console.error("Clustering failed:", e);
      setError(`Error during clustering: ${formatApiError(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAnyway = async () => {
    setAwaitingConfirmation(false);
    setError(null);
    setLoading(true);
    try {
      await executeRun(collectRunSettings());
    } catch (e) {
      console.error("Clustering failed:", e);
      setError(`Error during clustering: ${formatApiError(e)}`);
    } finally {
      setLoading(false);
    }
  };

  // Parameter sweep: the trials use the current settings with the swept values replaced.
  const handleStartSweep = async (sweepConfig) => {
    setError(null);
    if (!validateInputs()) return null;
    return postWithDataset(`/sessions/${sessionId}/sweeps`, collectRunSettings(), { sweep: JSON.stringify(sweepConfig) });
  };

  const refreshRuns = async () => {
    try {
      const data = await fetchJson(`${API_BASE}/sessions/${sessionId}/runs`);
      setRuns(data.runs);
    } catch (e) {
      console.error('Refreshing the run list failed:', e);
    }
  };

  const handleCheckOnly = async () => {
    setError(null);
    setAwaitingConfirmation(false);
    if (!validateInputs()) return;

    setLoading(true);
    try {
      await runPreflight(collectRunSettings());
    } catch (e) {
      setError(`Preflight check failed: ${formatApiError(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const handleExplore = async () => {
    setError(null);
    if (!validateInputs()) return;
    setLoading(true);
    try {
      setExploreData(await postWithDataset(`/sessions/${sessionId}/explore`, collectRunSettings()));
    } catch (e) {
      setError(`Exploratory analysis failed: ${formatApiError(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunUpdated = (updatedRun) => {
    setRuns(previousRuns => previousRuns.map(run => (run.run_id === updatedRun.run_id ? updatedRun : run)));
    // A finished stability analysis adds row stability to the displayed rows of the active run.
    if (updatedRun.run_id === activeRunId && updatedRun.stability && updatedRun.links?.result) {
      fetchJson(`${API_BASE}${updatedRun.links.result}`)
        .then(data => setClusterData(data.cluster_data))
        .catch(() => {});  // the table simply keeps its previous columns
    }
  };

  const handleLoadExample = async () => {
    const example = examples.find(item => item.id === selectedExampleId);
    if (!example) return;

    setError(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/examples/${example.id}/file`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const blob = await response.blob();
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      await loadSelectedFile(new File([blob], example.filename, { type: 'text/tab-separated-values' }));
      setNameColumn(example.name_column);
      setIntensityStartIndex(example.intensity_start_index);
      setPreprocessingOptions({ ...DEFAULT_PREPROCESSING, ...(example.preprocessing || {}) });
    } catch (e) {
      setError(`Could not load example data: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const computeCentroidsAndEdges = (data, clusterColors, profiles) => {
    // Cluster sizes from the full data; the points may be a sample of a large dataset.
    const fullSizes = {};
    if (profiles) profiles.clusters.forEach((cluster, index) => { fullSizes[cluster] = profiles.sizes[index]; });
    const clusters = {};
    data.forEach(d => {
      if (!clusters[d.cluster]) clusters[d.cluster] = [];
      clusters[d.cluster].push([d.x, d.y]);
    });
  
    const centroids = Object.entries(clusters).map(([cluster, points]) => {
      const meanX = points.reduce((sum, p) => sum + p[0], 0) / points.length;
      const meanY = points.reduce((sum, p) => sum + p[1], 0) / points.length;
      return { cluster: parseInt(cluster), x: meanX, y: meanY, size: fullSizes[cluster] ?? points.length };
    });
  
    const maxClusterSize = Math.max(...centroids.map(c => c.size));
    const minCentroidSize = 10; // Minimum size for a centroid
    const maxCentroidSize = 30; // Maximum size for a centroid
    const getCentroidSize = (clusterSize) => {
      if (maxClusterSize === 0) return minCentroidSize;
      return minCentroidSize + (maxCentroidSize - minCentroidSize) * (clusterSize / maxClusterSize);
    };

    const edges = [];
    for (let i = 0; i < centroids.length; i++) {
      for (let j = i + 1; j < centroids.length; j++) {
        const c1 = centroids[i];
        const c2 = centroids[j];
        const dist = Math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2);
        edges.push({ source: c1, target: c2, distance: dist });
      }
    }
  
    const maxNetworkDistance = Math.max(...edges.map(e => e.distance));

    const edgeTraces = edges.map((edge, idx) => {
      const safeDistance = Math.max(edge.distance, 0.01);
      const lineWidth = Math.min(3, 1 + 2 / safeDistance); // Existing width logic
      
      const normalizedDistance = edge.distance / maxNetworkDistance;
      const opacity = Math.max(0.2, 1 - normalizedDistance * 0.8);
      const colorValue = Math.floor(255 * normalizedDistance);
      const edgeColor = `rgba(${colorValue}, ${colorValue}, ${colorValue}, ${opacity})`; // Greyscale variant, or pick another color

      return {
        type: 'scatter',
        mode: 'lines',
        x: [edge.source.x, edge.target.x],
        y: [edge.source.y, edge.target.y],
        line: {
          width: lineWidth,
          color: edgeColor, 
        },
        hoverinfo: 'text',
        text: `Cluster ${edge.source.cluster} ↔ Cluster ${edge.target.cluster}<br>Distance: ${edge.distance.toFixed(3)}`,
        showlegend: false
      };
    });
  
    const nodeTrace = {
      type: 'scatter',
      mode: 'markers+text', 
      x: centroids.map(c => c.x),
      y: centroids.map(c => c.y),
      marker: {
        size: centroids.map(c => getCentroidSize(c.size)), 
        color: centroids.map(c => clusterColors[c.cluster] || '#000'), 
        line: { width: 1, color: '#000' }
      },
      text: centroids.map(c => `Cluster ${c.cluster}`), 
      textposition: 'top center',
      hoverinfo: 'text',

      hovertext: centroids.map(c => `Cluster ${c.cluster}<br>Candidates: ${c.size}`),
      name: 'Centroids'
    };
  
    return {
      data: [...edgeTraces, nodeTrace],
      layout: {
        title: 'Network of Cluster Centroids',
        xaxis: { title: 'X', showgrid: false, zeroline: false, showticklabels: true },
        yaxis: { title: 'Y', showgrid: false, zeroline: false, showticklabels: true },
        hovermode: 'closest',
        height: 700,
        autosize: true,
        plot_bgcolor: '#fcfcfc',
        paper_bgcolor: '#fcfcfc',
      }
    };
  };

  const handleSubnetworkClick = (clusterId, clusterColors) => {
    if (!clusterData) return;
  
    const nodes = clusterData.filter(d => d.cluster === clusterId);
    const edges = [];
  
    nodes.forEach((n1, i) => {
      const distances = nodes
        .map((n2, j) => {
          if (i === j) return null;
          const dist = Math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2);
          return { source: n1, target: n2, distance: dist };
        })
        .filter(Boolean)
        .sort((a, b) => a.distance - b.distance)
        .slice(0, topNNeighbors); // <== Use user-selected value

    });

    const nodeDegrees = {};
    nodes.forEach(node => (nodeDegrees[node.name] = 0)); // Initialize all node degrees to 0

    nodes.forEach((n1, i) => {
      const distances = nodes
        .map((n2, j) => {
          if (i === j) return null;
          const dist = Math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2);
          return { source: n1, target: n2, distance: dist };
        })
        .filter(Boolean)
        .sort((a, b) => a.distance - b.distance)
        .slice(0, topNNeighbors);

      // Update degrees for connected nodes
      distances.forEach(edge => {
          edges.push(edge); // Keep this line as it's part of edge creation
          nodeDegrees[edge.source.name]++;
          nodeDegrees[edge.target.name]++;
      });
    });
    
    const maxDegree = Math.max(...Object.values(nodeDegrees));
    const minNodeSize = 8; // Minimum size for a node
    const maxNodeSize = 25; // Maximum size for a node
    const getNodeSize = (degree) => {
      if (maxDegree === 0) return minNodeSize; // Avoid division by zero if all degrees are 0
      return minNodeSize + (maxNodeSize - minNodeSize) * (degree / maxDegree);
    };

    const maxDistanceInSubnetwork = () => {
      if (edges.length === 0) return 1; 
      return Math.max(...edges.map(e => e.distance));
    };
    const dynamicMaxDist = maxDistanceInSubnetwork() * 1.2; // Add a buffer
  
    const edgeTraces = edges.map(edge => {
      const safeDistance = Math.max(edge.distance, 0.01);
      const lineWidth = Math.min(3, 1 + 2 / safeDistance);
  
      const opacity = Math.max(0.2, 1 - (edge.distance / dynamicMaxDist));
      const edgeColor = `rgba(150, 150, 150, ${opacity})`;
  
      return {
        type: 'scatter',
        mode: 'lines',
        x: [edge.source.x, edge.target.x],
        y: [edge.source.y, edge.target.y],
        line: {
          width: lineWidth,
          color: edgeColor, // Use the calculated edgeColor
        },
        hoverinfo: 'text',
        text: `${edge.source.name} ↔ ${edge.target.name}<br>Distance: ${edge.distance.toFixed(3)}`,
        showlegend: false
      };
    });


    const nodeTrace = {
      type: 'scatter',
      mode: 'markers',
      x: nodes.map(n => n.x),
      y: nodes.map(n => n.y),
      marker: {
        size: nodes.map(n => getNodeSize(nodeDegrees[n.name])), // Variable node size based on degree
        color: clusterColors[clusterId] || '#2b8cbe',// Keep uniform for now, or introduce sub-cluster colors from 'clusterColors' if needed
        line: { width: 1, color: '#ffffff' },
        opacity: 0.9
      },
      text: '',
      textposition: 'top center',
      textfont: {
        size: 10,
        color: '#111'
      },
      hovertext: nodes.map(n => `Name: ${n.name}<br>Cluster: ${n.cluster}<br>Connections: ${nodeDegrees[n.name]}`),
      hoverinfo: 'text',
      name: `Cluster ${clusterId}`
    };
  
    setSubnetworkData([...edgeTraces, nodeTrace]);
    setSubnetworkLayout({
      title: `Sub-network of Cluster ${clusterId} (Top ${topNNeighbors} Connections)`,
      height: 700,
      autosize: true,
      xaxis: { visible: false },
      yaxis: { visible: false },
      hovermode: 'closest',
      plot_bgcolor: '#fcfcfc',
      paper_bgcolor: '#fcfcfc',
      margin: { l: 20, r: 20, t: 40, b: 20 },
      dragmode: 'lasso', // or select
    });
    setSelectedSubnetwork(clusterId);
  };

  const handlePointsSelected = async (points, clusterColors) => {
    if (!points || !points.points || points.points.length === 0) {
        setSelectedProfilesData([]);
        setSelectedProfilesLayout({});
        setSelectedProfileNames([]);
        return;
    }


    const selectedNames = points.points.map(p => p.hovertext.split('<br>')[0].replace('Name: ', '')); // Extract name from hovertext
    const selectedNodes = clusterData.filter(d => selectedNames.includes(d.name) && d.cluster === selectedSubnetwork);

    // Raw log10 intensities are fetched on demand to keep the main result small.
    const intensitiesByName = {};
    try {
        const data = await fetchJson(`${API_BASE}${activeRun.links.profiles}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ names: selectedNodes.map(node => node.name) }),
        });
        data.profiles.forEach(profile => { intensitiesByName[profile.name] = profile.intensities; });
    } catch (e) {
        setError(`Could not load the selected profiles: ${formatApiError(e)}`);
        return;
    }

    const linePlotTraces = [];
    const namesForCaption = [];

    selectedNodes.forEach(node => {
        const intensities = intensitiesByName[node.name];
        if (intensities && intensities.length > 0) {
            const numDimensions = intensities.length;
            linePlotTraces.push({
                x: Array.from({ length: numDimensions }, (_, i) => `${i + 1}`),
                y: intensities,
                mode: 'lines',
                name: node.name,
                line: { color: clusterColors[node.cluster], width: 1.5 }, // Use cluster color or a consistent color
                hoverinfo: 'name+y' // Show name and intensity value on hover
            });
            namesForCaption.push(node.name);
        }
    });

    setSelectedProfilesData(linePlotTraces);
    setSelectedProfileNames(namesForCaption);
    setSelectedProfilesLayout({
        title: `Intensity Profiles of Selected Candidates from Cluster ${selectedSubnetwork}`,
        xaxis: { title: 'Dimension' },
        yaxis: { title: 'Intensity/Value' },
        hovermode: 'closest',
        height: 500,
        autosize: true,
        plot_bgcolor: '#fcfcfc',
        paper_bgcolor: '#fcfcfc',
        showlegend: true // Show legend to identify individual lines
    });
};
  const plotData = clusterData ? [{
    x: clusterData.map(d => d.x),
    y: clusterData.map(d => d.y),
    mode: 'markers',
    type: 'scatter',
    marker: {
      color: clusterData.map(d => d.color),
      size: 7,
      line: {
        color: 'rgba(0, 0, 0, 0.5)',
        width: 0.5
      }
    },
    text: clusterData.map(d => `Name: ${d.name}<br>Cluster: ${d.cluster}<br>X: ${d.x.toFixed(3)}<br>Y: ${d.y.toFixed(3)}`),
    hoverinfo: 'text'
  }] : [];

  // Cluster Details table (and its TSV download) follow the column the user sorted by; ties fall back to name.
  const sortedClusterData = useMemo(() => {
    if (!clusterData) return [];
    const compareText = (a, b) => a.localeCompare(b, undefined, { sensitivity: 'base', numeric: true });
    const comparators = {
      name: (a, b) => compareText(a.name, b.name),
      cluster: (a, b) => a.cluster - b.cluster,
      x: (a, b) => a.x - b.x,
      y: (a, b) => a.y - b.y,
      color: (a, b) => compareText(a.color, b.color),
      membership: (a, b) => (a.membership ?? -1) - (b.membership ?? -1),
      stability: (a, b) => (a.stability ?? -1) - (b.stability ?? -1),
    };
    const sign = detailsSort.direction === 'asc' ? 1 : -1;
    return [...clusterData].sort((a, b) => (
      sign * comparators[detailsSort.key](a, b) || compareText(a.name, b.name)
    ));
  }, [clusterData, detailsSort]);

  // Optional per-row columns: membership (HDBSCAN, mixture models) and row stability (after a stability analysis).
  const hasMembership = Boolean(clusterData?.some(d => d.membership !== undefined));
  const hasRowStability = Boolean(clusterData?.some(d => d.stability !== undefined));
  const formatShare = (value) => (value === undefined || value === null ? '–' : value.toFixed(2));

  const toggleDetailsSort = (key) => {
    setDetailsSort(previous => ({
      key,
      direction: previous.key === key && previous.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  const renderSortableHeader = (key, label, style) => {
    const isSorted = detailsSort.key === key;
    return (
      <th
        className="sortable-header"
        style={style}
        onClick={() => toggleDetailsSort(key)}
        aria-sort={isSorted ? (detailsSort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
        title={`Sort by ${label}`}
      >
        {label}
        <span className="sort-indicator">{isSorted ? (detailsSort.direction === 'asc' ? ' ▲' : ' ▼') : ' ⇅'}</span>
      </th>
    );
  };

  const plotLayout = {
    title: `Clustering Results (${selectedAlgorithm} with ${selectedReduction})`,
    xaxis: { title: 'Dimension 1' },
    yaxis: { title: 'Dimension 2' },
    hovermode: 'closest',
    height: 700,
    autosize: true,
    margin: { l: 50, r: 50, b: 50, t: 50 },
    plot_bgcolor: '#f5f5f5',
    paper_bgcolor: '#f5f5f5',
    shapes: [],
  };

  const getProfileValues = useCallback((node) => {
    if (Array.isArray(node.profile_values) && node.profile_values.length > 0) {
      return node.profile_values;
    }
    return Array.isArray(node.intensities) ? node.intensities : [];
  }, []);

  const heatmapColumnLabels = useMemo(() => {
    const firstProfile = clusterData?.find(d => getProfileValues(d).length > 0);
    const profileLength = firstProfile ? getProfileValues(firstProfile).length : 0;
    if (intensityColumns.length === profileLength) {
      return intensityColumns;
    }
    return Array.from({ length: profileLength }, (_, i) => `Dimension ${i + 1}`);
  }, [clusterData, intensityColumns, getProfileValues]);

  const clusterSizes = useMemo(() => {
    const sizes = {};
    if (clusterProfiles) {
      clusterProfiles.clusters.forEach((cluster, index) => { sizes[cluster] = clusterProfiles.sizes[index]; });
    } else {
      (clusterData || []).forEach(row => { sizes[row.cluster] = (sizes[row.cluster] || 0) + 1; });
    }
    return sizes;
  }, [clusterProfiles, clusterData]);

  const heatmapClusters = useMemo(() => {
    if (!clusterData) return [];
    return Array.from(new Set(clusterData.map(d => d.cluster))).sort((a, b) => a - b);
  }, [clusterData]);

  const aggregateProfileValue = useCallback((rows, columnIndex) => {
    const values = rows
      .map(row => getProfileValues(row)[columnIndex])
      .filter(value => Number.isFinite(value));

    if (values.length === 0) return null;

    if (heatmapAggregation === 'mean') {
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }

    const sortedValues = [...values].sort((a, b) => a - b);
    const midpoint = Math.floor(sortedValues.length / 2);
    if (sortedValues.length % 2 === 0) {
      return (sortedValues[midpoint - 1] + sortedValues[midpoint]) / 2;
    }
    return sortedValues[midpoint];
  }, [heatmapAggregation, getProfileValues]);

  const profileDistance = useCallback((a, b) => {
    let sumSquares = 0;
    let count = 0;

    for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
      if (Number.isFinite(a[i]) && Number.isFinite(b[i])) {
        sumSquares += (a[i] - b[i]) ** 2;
        count += 1;
      }
    }

    return count === 0 ? Number.POSITIVE_INFINITY : Math.sqrt(sumSquares);
  }, []);

  const buildDendrogram = useCallback((rows) => {
    if (rows.length < 2) {
      return null;
    }

    const pairwiseDistances = new Map();
    const getPairKey = (a, b) => `${Math.min(a, b)}-${Math.max(a, b)}`;

    for (let i = 0; i < rows.length; i += 1) {
      for (let j = i + 1; j < rows.length; j += 1) {
        pairwiseDistances.set(getPairKey(i, j), profileDistance(rows[i].values, rows[j].values));
      }
    }

    const clusterDistance = (a, b) => {
      let total = 0;
      let count = 0;

      a.members.forEach(i => {
        b.members.forEach(j => {
          const distance = pairwiseDistances.get(getPairKey(i, j));
          if (Number.isFinite(distance)) {
            total += distance;
            count += 1;
          }
        });
      });

      return count === 0 ? Number.POSITIVE_INFINITY : total / count;
    };

    let nextId = rows.length;
    let clusters = rows.map((row, index) => ({
      id: index,
      members: [index],
      row,
      left: null,
      right: null,
      distance: 0
    }));

    while (clusters.length > 1) {
      let bestI = 0;
      let bestJ = 1;
      let bestDistance = Number.POSITIVE_INFINITY;

      for (let i = 0; i < clusters.length; i += 1) {
        for (let j = i + 1; j < clusters.length; j += 1) {
          const distance = clusterDistance(clusters[i], clusters[j]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestI = i;
            bestJ = j;
          }
        }
      }

      if (!Number.isFinite(bestDistance)) {
        break;
      }

      const left = clusters[bestI];
      const right = clusters[bestJ];
      const merged = {
        id: nextId,
        members: [...left.members, ...right.members],
        row: null,
        left,
        right,
        distance: bestDistance
      };
      nextId += 1;

      clusters = clusters.filter((_, index) => index !== bestI && index !== bestJ);
      clusters.push(merged);
    }

    const root = clusters[0];
    const leafOrder = [];
    const collectLeaves = (node) => {
      if (!node.left && !node.right) {
        leafOrder.push(node.id);
        return;
      }
      collectLeaves(node.left);
      collectLeaves(node.right);
    };
    collectLeaves(root);

    const yByLeaf = new Map(leafOrder.map((leafId, index) => [leafId, index]));
    const getNodeY = (node) => {
      if (!node.left && !node.right) {
        return yByLeaf.get(node.id);
      }
      return (getNodeY(node.left) + getNodeY(node.right)) / 2;
    };

    const traces = [];
    const addTraces = (node) => {
      if (!node.left || !node.right) return;

      const leftY = getNodeY(node.left);
      const rightY = getNodeY(node.right);

      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [node.left.distance, node.distance, node.distance, node.right.distance],
        y: [leftY, leftY, rightY, rightY],
        line: { color: '#444', width: 1.2 },
        hoverinfo: 'skip',
        showlegend: false
      });

      addTraces(node.left);
      addTraces(node.right);
    };
    addTraces(root);

    return {
      orderedRows: leafOrder.map(index => rows[index]),
      traces,
      maxDistance: root.distance || 1
    };
  }, [profileDistance]);

  const heatmapSummaryRows = useMemo(() => {
    if (!clusterData || heatmapColumnLabels.length === 0) return [];

    // Summaries computed on the server from all rows (the points may be a sample).
    if (clusterProfiles) {
      const values = heatmapAggregation === 'mean' ? clusterProfiles.mean : clusterProfiles.median;
      return clusterProfiles.clusters.map((clusterId, index) => ({
        clusterId,
        label: `${clusterId === -1 ? 'Noise (-1)' : `Cluster ${clusterId}`}, n=${clusterProfiles.sizes[index]}`,
        color: clusterColors[clusterId] || '#808080',
        values: values[index],
      }));
    }

    return heatmapClusters.map(clusterId => {
      const rows = clusterData.filter(row => row.cluster === clusterId);
      return {
        clusterId,
        label: clusterId === -1 ? `Noise (-1), n=${rows.length}` : `Cluster ${clusterId}, n=${rows.length}`,
        color: clusterColors[clusterId] || '#808080',
        values: heatmapColumnLabels.map((_, columnIndex) => aggregateProfileValue(rows, columnIndex))
      };
    });
  }, [clusterData, clusterProfiles, heatmapAggregation, clusterColors, heatmapClusters, heatmapColumnLabels, aggregateProfileValue]);

  const selectedHeatmapRows = useMemo(() => {
    if (!clusterData || heatmapCluster === 'summary') return [];

    return clusterData
      .filter(row => row.cluster === Number(heatmapCluster))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
  }, [clusterData, heatmapCluster]);

  const visibleHeatmapRows = selectedHeatmapRows.slice(0, heatmapRowLimit);
  const dendrogramRowLimit = 250;
  const heatmapValueLabel = heatmapCluster === 'summary'
    ? `${heatmapAggregation === 'mean' ? 'Mean' : 'Median'} value (${valueLabel})`
    : `Value (${valueLabel})`;
  const heatmapRows = useMemo(() => (
    heatmapCluster === 'summary'
      ? heatmapSummaryRows
      : visibleHeatmapRows.map(row => ({
          label: row.name,
          values: getProfileValues(row)
        }))
  ), [heatmapCluster, heatmapSummaryRows, visibleHeatmapRows, getProfileValues]);
  const heatmapDendrogram = useMemo(() => {
    if (!showHeatmapDendrogram || heatmapRows.length > dendrogramRowLimit) {
      return null;
    }
    return buildDendrogram(heatmapRows);
  }, [showHeatmapDendrogram, heatmapRows, buildDendrogram]);
  const orderedHeatmapRows = heatmapDendrogram?.orderedRows || heatmapRows;
  const hasVisibleDendrogram = Boolean(heatmapDendrogram);
  const heatmapRowPixelHeight = heatmapCluster === 'summary' ? 38 : 18;
  const heatmapVisibleRowCount = heatmapCluster === 'summary'
    ? heatmapSummaryRows.length
    : visibleHeatmapRows.length;
  const heatmapLabelFontSize = heatmapCluster === 'summary'
    ? 12
    : heatmapVisibleRowCount > 500
      ? 7
      : heatmapVisibleRowCount > 250
        ? 8
        : heatmapVisibleRowCount > 120
          ? 9
          : 10;
  const longestHeatmapLabelLength = orderedHeatmapRows.reduce(
    (maxLength, row) => Math.max(maxLength, String(row.label || '').length),
    0
  );
  const heatmapLeftMargin = hasVisibleDendrogram
    ? 40
    : heatmapCluster === 'summary'
      ? 170
      : Math.min(360, Math.max(220, longestHeatmapLabelLength * heatmapLabelFontSize * 0.58 + 34));
  const heatmapHeight = heatmapCluster === 'summary'
    ? Math.max(380, heatmapSummaryRows.length * heatmapRowPixelHeight + 170)
    : Math.max(520, visibleHeatmapRows.length * heatmapRowPixelHeight + 190);
  const heatmapRowPositions = orderedHeatmapRows.map((_, index) => index);
  const heatmapTrace = {
    type: 'heatmap',
    z: orderedHeatmapRows.map(row => row.values),
    x: heatmapColumnLabels,
    y: heatmapRowPositions,
    xaxis: hasVisibleDendrogram ? 'x2' : 'x',
    yaxis: 'y',
    customdata: orderedHeatmapRows.map(row => row.values.map(() => row.label)),
    colorscale: 'RdBu',
    reversescale: true,
    zmid: 0,
    colorbar: { title: heatmapValueLabel },
    hovertemplate: `%{customdata}<br>%{x}: %{z:.3f}<br>${heatmapValueLabel}<extra></extra>`
  };
  const heatmapData = hasVisibleDendrogram
    ? [
        ...heatmapDendrogram.traces.map(trace => ({
          ...trace,
          xaxis: 'x',
          yaxis: 'y'
        })),
        heatmapTrace
      ]
    : [heatmapTrace];

  const heatmapLayout = {
    title: heatmapCluster === 'summary'
      ? `Cluster ${heatmapAggregation === 'mean' ? 'Mean' : 'Median'} Profiles`
      : `Cluster ${heatmapCluster} Profiles`,
    height: heatmapHeight,
    autosize: true,
    margin: {
      l: heatmapLeftMargin,
      r: 40,
      t: 70,
      b: 120
    },
    xaxis: {
      title: hasVisibleDendrogram ? 'Distance' : 'Intensity columns',
      domain: hasVisibleDendrogram ? [0, 0.16] : [0, 1],
      showgrid: false,
      zeroline: false,
      range: hasVisibleDendrogram ? [Math.max(heatmapDendrogram?.maxDistance || 1, 1e-6), 0] : undefined,
      automargin: true,
      tickangle: hasVisibleDendrogram ? 0 : (heatmapColumnLabels.length > 12 ? -45 : 0)
    },
    ...(hasVisibleDendrogram ? {
      xaxis2: {
        title: 'Intensity columns',
        domain: [0.31, 1],
        anchor: 'y',
        automargin: true,
        tickangle: heatmapColumnLabels.length > 12 ? -45 : 0
      }
    } : {}),
    yaxis: {
      title: heatmapCluster === 'summary' ? 'Clusters' : nameColumn,
      anchor: hasVisibleDendrogram ? 'x2' : 'x',
      side: 'left',
      tickmode: 'array',
      tickvals: heatmapRowPositions,
      ticktext: orderedHeatmapRows.map(row => row.label),
      range: [Math.max(orderedHeatmapRows.length - 0.5, 0.5), -0.5],
      automargin: true,
      tickfont: { size: heatmapLabelFontSize },
      ticklabeloverflow: 'allow',
      showticklabels: true
    },
    plot_bgcolor: '#fcfcfc',
    paper_bgcolor: '#fcfcfc',
    showlegend: false
  };

  const downloadMethodDetails = () => {
    if (!methodDetails.trim()) {
      alert("No method details available to download.");
      return;
    }
  
    const blob = new Blob([methodDetails], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `method_details_${selectedAlgorithm}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // --- Helper to clear file input ---
  const clearFileInput = () => {
    setFile(null);
    datasetIdRef.current = null;
    setPreflightReport(null);
    setExploreData(null);
    setAwaitingConfirmation(false);
    setFileName('');
    setFileType('');
    setColumnNames([]);
    setExcelSheetNames([]);
    setExcelColumnsBySheet({});
    setSelectedSheetName('');
    setNameColumn('Name');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getFormattedParamText = (definitions, customParams, useDefaults) => (
    definitions
      .map(param => {
        const value = useDefaults ? param.default : (customParams[param.name] ?? param.default);
        return `${param.name}=${value}`;
      })
      .join(', ')
  );

  const clusteringParamText = selectedAlgorithm
    ? getFormattedParamText(algoParams[selectedAlgorithm] || [], customAlgorithmParams, useDefaultAlgorithmParams)
    : '';

  const reductionParamText = selectedReduction !== 'None'
    ? getFormattedParamText(
        reductionParamsDef[selectedReduction] || [],
        customReductionParams,
        useDefaultReductionParams
      )
    : '';

  const methodsTemplateText = selectedAlgorithm
    ? selectedReduction === 'None'
      ? `Clustering was applied without dimensionality reduction using ${selectedAlgorithm}${
          clusteringParamText ? ` with the following parameters: ${clusteringParamText}` : ''
        }.`
      : `The data were first reduced in dimensionality using ${selectedReduction}${
          reductionParamText ? ` with the following parameters: ${reductionParamText}` : ''
        }, and then clustered using ${selectedAlgorithm}${
          clusteringParamText ? ` with the following parameters: ${clusteringParamText}` : ''
        }.`
    : '';

  return (
    <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
        <a
          href="https://www.rappsilberlab.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img
            src={require('./RapLabTextLogo.png')}
            alt="RapLab Logo"
            style={{ height: '90px', marginRight: '30px' }}
          />
        </a>
        <img
          src={require('./logo_rapcluster.svg').default}
          alt="RapCluster Logo"
          style={{ height: '90px', marginRight: '40px' }}
        />
        </div>
          <h1 style={{ flexGrow: 1, textAlign: 'center', margin: 0 }}>Clustering at Your Fingertips</h1>
          <div style={{ width: '100px' }} /> {/* filler to balance the logo space */}
        </div>
      </header>
      <main className="App-main">
        <form onSubmit={handleSubmit} className="clustering-form">
          <section className="input-section">
          <h2>
              1. Upload Data<HelpTip text={HELP.upload} label="Upload data" />
              <a
                href="https://raw.githubusercontent.com/lutfia95/RapCluster/main/example_data.tsv"
                download="example_data.tsv"
                style={{ fontSize: '0.7em', marginLeft: '8px' }}
              >
                (example data)
              </a>
            </h2>
            <div className="form-group">
              <label htmlFor="file-upload">Select File (CSV/TSV/Excel):<HelpTip text={HELP.upload} label="file upload" /></label>
              <input 
                type="file" 
                id="file-upload" 
                accept=".csv,.tsv,.xlsx,.xls" 
                onChange={handleFileChange} 
                ref={fileInputRef}
              />
              {fileName && <span className="file-name">{fileName} <button type="button" onClick={clearFileInput}>x</button></span>}
            </div>

            {examples.length > 0 && (
              <div className="form-group">
                <label htmlFor="example-select">Or load example data:<HelpTip text={HELP.example} label="example data" /></label>
                <div className="example-loader">
                  <select
                    id="example-select"
                    value={selectedExampleId}
                    onChange={(e) => setSelectedExampleId(e.target.value)}
                  >
                    <option value="">Choose an example…</option>
                    {examples.map(example => (
                      <option key={example.id} value={example.id}>{example.title}</option>
                    ))}
                  </select>
                  <button type="button" onClick={handleLoadExample} disabled={!selectedExampleId || loading}>
                    Load
                  </button>
                </div>
                {selectedExampleId && (
                  <span className="example-description">
                    {examples.find(example => example.id === selectedExampleId)?.description}
                  </span>
                )}
              </div>
            )}

            {fileType === 'xlsx' && excelSheetNames.length > 0 && (
              <div className="form-group">
                <label htmlFor="sheet-selector">Sheet:<HelpTip text={HELP.sheet} label="sheet" /></label>
                <select
                  id="sheet-selector"
                  value={selectedSheetName}
                  onChange={(e) => handleExcelSheetChange(e.target.value)}
                  required
                >
                  {excelSheetNames.map(sheetName => (
                    <option key={sheetName} value={sheetName}>{sheetName}</option>
                  ))}
                </select>
              </div>
            )}

            <div className="form-group">
              <label htmlFor="name-column">Name of Node Column:<HelpTip text={HELP.nameColumn} label="name column" /></label>
              {['csv', 'tsv', 'xlsx'].includes(fileType) && columnNames.length > 0 ? (
                <select
                  id="name-column"
                  value={columnNames.includes(nameColumn) ? nameColumn : columnNames[0]}
                  onChange={(e) => handleNameColumnChange(e.target.value)}
                  required
                >
                  {columnNames.map(columnName => (
                    <option key={columnName} value={columnName}>{columnName}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  id="name-column"
                  value={nameColumn}
                  onChange={(e) => handleNameColumnChange(e.target.value)}
                  placeholder="e.g., GeneName"
                  required
                />
              )}
            </div>

            <div className="form-group">
              <label htmlFor="intensity-start-index">Data Start Column Index (0-based):<HelpTip text={HELP.startIndex} label="data start column" /></label>
              <input
                type="number"
                id="intensity-start-index"
                value={intensityStartIndex}
                onChange={(e) => setIntensityStartIndex(parseInt(e.target.value))}
                min="0"
                required
              />
            </div>
          </section>

          <PreprocessingOptions
            options={preprocessingOptions}
            onChange={setPreprocessingOptions}
            onExplore={handleExplore}
            disabled={loading || !file}
          />

          <section className="reduction-section">
            <h2>3. Dimensionality Reduction<HelpTip text={HELP.reduction} label="dimensionality reduction" /></h2>
             <p style={{ marginTop: '-10px' }}>
                <button
                  type="button"
                  onClick={() => setShowReductionModal(true)}
                  title={HELP.learnMore}
                  style={{
                    fontSize: '0.9em',
                    background: 'none',
                    color: '#007bff',
                    border: 'none',
                    cursor: 'pointer',
                    paddingLeft: '0'
                  }}
                >
                  Learn More
                </button>
              </p>
            <div className="form-group">
              <label htmlFor="reduction-method">
                Method:<HelpTip text={REDUCTION_HELP[selectedReduction]} label={selectedReduction} />
              </label>
              <select
                id="reduction-method"
                value={selectedReduction}
                onChange={(e) => {
                  setSelectedReduction(e.target.value);
                  setUseDefaultReductionParams(true); // Reset to default when method changes
                  setCustomReductionParams({});
                }}
              >
                {availableReductions.map(method => (
                  <option key={method} value={method} title={REDUCTION_HELP[method]}>{method}</option>
                ))}
              </select>
              <small className="method-hint">{methodDescriptions.reduction[selectedReduction]}</small>
            </div>

            {selectedReduction !== 'None' && (
              <div className="param-control">
                <label>
                  <input
                    type="checkbox"
                    checked={useDefaultReductionParams}
                    onChange={(e) => {
                        setUseDefaultReductionParams(e.target.checked);
                        if(e.target.checked) setCustomReductionParams({}); // Clear custom if defaulting
                    }}
                  />
                  Use Default Parameters<HelpTip text={HELP.useDefaults} label="default parameters" />
                </label>
                {!useDefaultReductionParams && (
                  <div className="param-custom-settings">
                    <h3>Custom {selectedReduction} Parameters</h3>
                    {renderParams(reductionParamsDef, customReductionParams, handleReductionParamChange, false, selectedReduction)}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="clustering-section">
            <h2>4. Clustering Algorithm<HelpTip text={HELP.clustering} label="clustering algorithm" /></h2>
            <p style={{ marginTop: '-10px' }}>
              <button
                type="button"
                onClick={() => setShowClusteringModal(true)}
                title={HELP.learnMore}
                style={{
                  fontSize: '0.9em',
                  background: 'none',
                  color: '#007bff',
                  border: 'none',
                  cursor: 'pointer',
                  paddingLeft: '0'
                }}
              >
                Learn More
              </button>
            </p>
            <div className="form-group">
              <label htmlFor="clustering-algorithm">
                Algorithm:<HelpTip text={ALGORITHM_HELP[selectedAlgorithm]} label={selectedAlgorithm} />
              </label>
              <select
                id="clustering-algorithm"
                value={selectedAlgorithm}
                onChange={(e) => {
                  setSelectedAlgorithm(e.target.value);
                  setUseDefaultAlgorithmParams(true); // Reset to default when algorithm changes
                  setCustomAlgorithmParams({});
                }}
                required
              >
                {availableAlgos.map(algo => (
                  <option key={algo} value={algo} title={ALGORITHM_HELP[algo]}>{algo}</option>
                ))}
              </select>
              {selectedAlgorithm && <small className="method-hint">{methodDescriptions.clustering[selectedAlgorithm]}</small>}
            </div>

            {selectedAlgorithm && (
              <div className="param-control">
                <label>
                  <input
                    type="checkbox"
                    checked={useDefaultAlgorithmParams}
                    onChange={(e) => {
                        setUseDefaultAlgorithmParams(e.target.checked);
                        if(e.target.checked) setCustomAlgorithmParams({}); // Clear custom if defaulting
                    }}
                  />
                  Use Default Parameters<HelpTip text={HELP.useDefaults} label="default parameters" />
                </label>
                {!useDefaultAlgorithmParams && (
                  <div className="param-custom-settings">
                    <h3>Custom {selectedAlgorithm} Parameters</h3>
                    {renderParams(algoParams, customAlgorithmParams, handleAlgorithmParamChange, false, selectedAlgorithm)}
                  </div>
                )}
              </div>
            )}
          </section>
          <p style={{ textAlign: 'left', marginTop: '16px', marginBottom: '0' }}>
              <a
                href="https://youtu.be/m7LynlO7T5U"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  fontSize: '0.95em',
                  color: '#007bff',
                  textDecoration: 'none'
                }}
              >
                Watch Youtube Tutorial
              </a>
            </p>

          <button type="button" className="check-button" onClick={handleCheckOnly} disabled={loading || !file} title={HELP.check}>
            Check data &amp; settings
          </button>
          <button type="submit" className="submit-button" disabled={loading || !file} title={HELP.run}>
            {loading ? 'Processing...' : 'Run Clustering'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {jobProgress && (
          <div className="job-progress" role="status">
            <span className="job-progress-spinner" aria-hidden="true" />
            <span className="job-progress-text">
              <strong>Run #{jobProgress.run.number} is running:</strong> {jobProgress.message || jobProgress.stage}
              <span className="job-progress-elapsed">
                {' '}({formatDuration(jobProgress.elapsed)} elapsed; runs are stopped after {formatDuration(jobProgress.run.limits.timeout_seconds)})
              </span>
            </span>
            <button type="button" onClick={handleCancelRun}>Cancel run</button>
          </div>
        )}

        <ExploratoryAnalysis data={exploreData} onClose={() => setExploreData(null)} />

        <ParameterSweep
          algorithm={selectedAlgorithm}
          paramDefs={algoParams[selectedAlgorithm]}
          reductionMethod={selectedReduction}
          canRun={Boolean(file)}
          apiBase={API_BASE}
          runs={runs}
          loading={loading}
          onStart={handleStartSweep}
          onRefreshRuns={refreshRuns}
          onView={handleViewRun}
        />

        <PreflightReport
          report={preflightReport}
          awaitingConfirmation={awaitingConfirmation}
          onRunAnyway={handleRunAnyway}
          onCancel={() => setAwaitingConfirmation(false)}
          onClose={() => { setPreflightReport(null); setAwaitingConfirmation(false); }}
          loading={loading}
        />

        <RunHistory
          runs={runs}
          activeRunId={activeRunId}
          apiBase={API_BASE}
          loading={loading}
          onView={handleViewRun}
          onLoadSettings={restoreRunSettings}
        />

        <Leaderboard
          runs={runs}
          apiBase={API_BASE}
          loading={loading}
          onView={handleViewRun}
          onRunUpdated={handleRunUpdated}
        />

        <RunComparison runs={runs} apiBase={API_BASE} sessionId={sessionId} />

        {metrics && clusterData && (
          <section className="results-section">
            <h2>
              Clustering Results
              {activeRun && ` — Run #${activeRun.number}`}
              <HelpTip text={HELP.results} label="results" />
            </h2>
            {activeRun && (
              <a className="download-all-button" href={`${API_BASE}${activeRun.links.download_zip}`} title={HELP.downloadZip}>
                Download All (ZIP)
              </a>
            )}
            {sampling?.sampled && (
              <div className="sampling-note">
                Large dataset: the plots and the table show {sampling.shown.toLocaleString()} of {sampling.total.toLocaleString()} rows
                (a sample stratified by cluster, so every cluster is included). Cluster sizes, the cluster summary heatmap and all
                downloads (TSV, ZIP) use all rows; the silhouette score is computed on a random sample of up to {(limits.silhouette_sample_size || 5000).toLocaleString()} rows.
              </div>
            )}
            {displayInfo && (
              <div className="compare-note display-note">
                Clustering and metrics used {displayInfo.clustering_dims} dimension(s); the scatter plot shows {displayInfo.description}.
              </div>
            )}
            <div className="metrics-summary">
              <h3>Metrics:</h3>
              <p><strong>Clusters Found:</strong><HelpTip text={HELP.clustersFound} label="clusters found" /> {metrics.n_clusters}</p>
              <p><strong>Silhouette Score:</strong><HelpTip text={HELP.silhouette} label="Silhouette score" /> {metrics.silhouette_score.toFixed(3)} (Higher is better)</p>
              <p><strong>Calinski-Harabasz Score:</strong><HelpTip text={HELP.calinskiHarabasz} label="Calinski-Harabasz score" /> {metrics.calinski_harabasz_score.toFixed(3)} (Higher is better)</p>
              <p><strong>Davies-Bouldin Score:</strong><HelpTip text={HELP.daviesBouldin} label="Davies-Bouldin score" /> {metrics.davies_bouldin_score.toFixed(3)} (Lower is better)</p>
              {Number.isFinite(metrics.dbcv) && (
                <p><strong>DBCV:</strong><HelpTip text={HELP.dbcv} label="DBCV" /> {metrics.dbcv.toFixed(3)} (Higher is better; density-based validation)</p>
              )}
              {Number.isFinite(metrics.bic) && (
                <p><strong>BIC / AIC:</strong><HelpTip text={HELP.bicAic} label="BIC and AIC" /> {metrics.bic.toFixed(1)} / {metrics.aic.toFixed(1)} (Lower is better; compare runs on the same data only)</p>
              )}
              {Number.isFinite(metrics.mean_membership) && (
                <p><strong>Mean membership:</strong><HelpTip text={HELP.meanMembership} label="mean membership" /> {metrics.mean_membership.toFixed(3)} (average assignment probability of clustered rows)</p>
              )}
              {activeRun?.model?.n_components_used !== undefined && (
                <p><strong>Components used:</strong><HelpTip text={HELP.componentsUsed} label="components used" /> {activeRun.model.n_components_used} of at most {activeRun.model.n_components_max}</p>
              )}
              <p><strong>Runtime:</strong><HelpTip text={HELP.runtime} label="runtime" /> {metrics.runtime_seconds.toFixed(2)} seconds</p>
            </div>

            <div className="visualization">
              <h3>Visualization:</h3>

              <div className="tab-buttons">
                <button
                  onClick={() => setActiveTab("scatter")}
                  title={HELP.tabScatter}
                  className={activeTab === "scatter" ? "active" : ""}
                >
                  Scatter Plot
                </button>
                <button
                  onClick={() => setActiveTab("network")}
                  title={HELP.tabNetwork}
                  className={activeTab === "network" ? "active" : ""}
                >
                  Clusters Network
                </button>
                <button
                  onClick={() => setActiveTab("subnetwork")}
                  title={HELP.tabSubnetwork}
                  className={activeTab === "subnetwork" ? "active" : ""}
                  disabled={!subnetworkData.length}
                >
                  Sub-network View
                </button>
                <button
                  onClick={() => setActiveTab("heatmap")}
                  title={HELP.tabHeatmap}
                  className={activeTab === "heatmap" ? "active" : ""}
                >
                  Heatmap
                </button>
                <button
                  onClick={() => setActiveTab("characterization")}
                  title={HELP.tabCharacterization}
                  className={activeTab === "characterization" ? "active" : ""}
                >
                  Characterization
                </button>
                <button
                  onClick={() => setActiveTab("stability")}
                  title={HELP.tabStability}
                  className={activeTab === "stability" ? "active" : ""}
                >
                  Stability
                </button>
              </div>

              {activeTab === "scatter" && (
                <>
                  <Plot
                    data={plotData}
                    layout={plotLayout}
                    useResizeHandler
                    style={{ width: '100%' }}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      toImageButtonOptions: {
                        format: "svg",
                        filename: "clustering_plot",
                        scale: 1,
                      },
                    }}
                  />
                  <div className="plot-caption">
                    <h4>Caption Template:</h4>
                    <p>
                      Clustering results using <strong>{selectedAlgorithm}</strong>{" "}
                      {useDefaultAlgorithmParams
                        ? "(default parameters)"
                        : `with custom parameters: ${JSON.stringify(customAlgorithmParams)}`}.{" "}
                      Dimensionality reduction: <strong>{selectedReduction}</strong>{" "}
                      {selectedReduction !== "None"
                        ? useDefaultReductionParams
                          ? "(default parameters)"
                          : `with custom parameters: ${JSON.stringify(
                              customReductionParams
                            )}`
                        : ""}. Evaluation scores: Silhouette:{" "}
                      <strong>{metrics.silhouette_score.toFixed(3)}</strong>, CH:{" "}
                      <strong>{metrics.calinski_harabasz_score.toFixed(3)}</strong>, Davies-Bouldin:{" "}
                      <strong>{metrics.davies_bouldin_score.toFixed(3)}</strong>.
                    </p>
                  </div>
                </>
              )}

              {activeTab === "network" && (
                <>
                  <Plot
                    data={networkPlotData}
                    layout={networkPlotLayout}
                    useResizeHandler
                    style={{ width: '100%' }}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      toImageButtonOptions: {
                        format: "svg",
                        filename: "network_plot",
                        scale: 1,
                      },
                    }}
                    onClick={(e) => {
                      const clicked = e.points[0];
                      if (clicked?.text?.startsWith("Cluster")) {
                        const clusterId = parseInt(clicked.text.replace("Cluster ", ""));
                        handleSubnetworkClick(clusterId, clusterColors);
                        setActiveTab("subnetwork");
                      }
                    }}
                  />

                  <div className="plot-caption">
                    <h4>Caption Template:</h4>
                    <p>
                      Network of cluster centroids generated from{" "}
                      <strong>{selectedAlgorithm}</strong>. Each edge represents inverse
                      Euclidean distance in 2D space derived via{" "}
                      <strong>{selectedReduction}</strong>. Total clusters:{" "}
                      <strong>{metrics.n_clusters}</strong>.
                    </p>
                  </div>
                </>
              )}

              {activeTab === "heatmap" && (
                <>
                  <div className="heatmap-explanation">
                    <strong>Values:</strong> {valueLabel}.
                    <span> The dendrogram is hierarchical ordering for the heatmap profiles, not a separate clustering algorithm.</span>
                  </div>

                  <div className="heatmap-controls">
                    <label>
                      View
                      <select
                        value={heatmapCluster}
                        onChange={(e) => setHeatmapCluster(e.target.value)}
                      >
                        <option value="summary">Cluster summary</option>
                        {heatmapClusters.map(clusterId => {
                          const count = clusterSizes[clusterId] ?? 0;
                          const label = clusterId === -1 ? `Noise (-1), n=${count}` : `Cluster ${clusterId}, n=${count}`;
                          return (
                            <option key={clusterId} value={String(clusterId)}>
                              {label}
                            </option>
                          );
                        })}
                      </select>
                    </label>

                    {heatmapCluster === 'summary' && (
                      <label>
                        Summary
                        <select
                          value={heatmapAggregation}
                          onChange={(e) => setHeatmapAggregation(e.target.value)}
                        >
                          <option value="median">Median</option>
                          <option value="mean">Mean</option>
                        </select>
                      </label>
                    )}

                    {heatmapCluster !== 'summary' && (
                      <label>
                        Rows
                        <select
                          value={heatmapRowLimit}
                          onChange={(e) => setHeatmapRowLimit(parseInt(e.target.value))}
                        >
                          {[100, 250, 500, 1000].map(limit => (
                            <option key={limit} value={limit}>{limit}</option>
                          ))}
                        </select>
                      </label>
                    )}

                    <label className="heatmap-checkbox">
                      <input
                        type="checkbox"
                        checked={showHeatmapDendrogram}
                        onChange={(e) => setShowHeatmapDendrogram(e.target.checked)}
                      />
                      Show hierarchical dendrogram
                    </label>
                  </div>

                  {heatmapTrace.z.length > 0 ? (
                    <div className="heatmap-plot-row">
                      {showHeatmapDendrogram && !heatmapDendrogram && (
                        <div className="dendrogram-placeholder">
                          {heatmapRows.length < 2
                            ? 'At least two rows are needed for the dendrogram.'
                            : `Reduce rows to ${dendrogramRowLimit} or fewer to show the dendrogram.`}
                        </div>
                      )}
                      <Plot
                        data={heatmapData}
                        layout={heatmapLayout}
                    useResizeHandler
                    style={{ width: '100%' }}
                        config={{
                          responsive: true,
                          displayModeBar: true,
                          toImageButtonOptions: {
                            format: "svg",
                            filename: heatmapCluster === 'summary' ? "cluster_summary_heatmap" : `cluster_${heatmapCluster}_heatmap`,
                            scale: 1,
                          },
                        }}
                      />
                    </div>
                  ) : (
                    <p>No heatmap values are available for this result.</p>
                  )}

                  {heatmapCluster !== 'summary' && selectedHeatmapRows.length > visibleHeatmapRows.length && (
                    <p className="heatmap-note">
                      Showing {visibleHeatmapRows.length} of {selectedHeatmapRows.length} rows in this cluster.
                    </p>
                  )}

                  {showHeatmapDendrogram && heatmapRows.length > dendrogramRowLimit && (
                    <p className="heatmap-note">
                      Dendrogram hidden for this view because it has {heatmapRows.length} rows. Reduce the row limit to {dendrogramRowLimit} or fewer to show it.
                    </p>
                  )}

                  <div className="plot-caption">
                    <h4>Caption Template:</h4>
                    <p>
                      Heatmap of <strong>preprocessed profiles ({valueLabel})</strong> after clustering with{" "}
                      <strong>{selectedAlgorithm}</strong>. Rows are{" "}
                      {heatmapCluster === 'summary'
                        ? `${heatmapAggregation} cluster profiles`
                        : `${nameColumn} entries from Cluster ${heatmapCluster}`}
                      ; columns are the uploaded intensity columns. The dendrogram, when shown, is hierarchical ordering of these heatmap profiles and is not a separate RapCluster clustering result.
                    </p>
                  </div>
                </>
              )}

              {activeTab === "characterization" && activeRun && (
                <ClusterCharacterization run={activeRun} apiBase={API_BASE} clusterColors={clusterColors} />
              )}

              {activeTab === "stability" && activeRun && (
                <StabilityAnalysis run={activeRun} apiBase={API_BASE} onRunUpdated={handleRunUpdated} />
              )}

              {activeTab === "subnetwork" && (
                <section>
                    <h2>Sub-network View</h2>
                    <div style={{ marginBottom: '10px' }}>
                        <button
                            onClick={() => setSubnetworkLayout(prev => ({ ...prev, dragmode: 'lasso' }))}
                            style={{
                                marginRight: '10px',
                                padding: '8px 15px',
                                cursor: 'pointer',
                                backgroundColor: subnetworkLayout.dragmode === 'lasso' ? '#28a745' : '#e8e8e8',
                                color: subnetworkLayout.dragmode === 'lasso' ? 'white' : '#333',
                                border: '1px solid #ccc',
                                borderRadius: '5px',
                                transition: 'background-color 0.15s ease-in-out'
                            }}
                        >
                            Select (Lasso)
                        </button>
                        <button
                            onClick={() => setSubnetworkLayout(prev => ({ ...prev, dragmode: 'pan' }))}
                            style={{
                                padding: '8px 15px',
                                cursor: 'pointer',
                                backgroundColor: subnetworkLayout.dragmode === 'pan' ? '#28a745' : '#e8e8e8',
                                color: subnetworkLayout.dragmode === 'pan' ? 'white' : '#333',
                                border: '1px solid #ccc',
                                borderRadius: '5px',
                                transition: 'background-color 0.15s ease-in-out'
                            }}
                        >
                            Pan/Zoom
                        </button>
                    </div>

                    <>
                        <div style={{ marginBottom: '10px', textAlign: 'center' }}>
                            <label>
                                Show top&nbsp;
                                <select
                                    value={topNNeighbors}
                                    onChange={(e) => setTopNNeighbors(parseInt(e.target.value))}
                                >
                                    {[3, 5, 10, 15, 20].map(n => (
                                        <option key={n} value={n}>{n}</option>
                                    ))}
                                </select>
                                &nbsp;nearest connections
                            </label>
                        </div>
                        <Plot
                            data={subnetworkData}
                            layout={subnetworkLayout}
                    useResizeHandler
                    style={{ width: '100%' }}
                            onSelected={(points) => handlePointsSelected(points, clusterColors)}
                            config={{
                                responsive: true,
                                displayModeBar: true,
                                toImageButtonOptions: {
                                    format: "svg",
                                    filename: `subnetwork_cluster_${selectedSubnetwork}`,
                                    scale: 1,
                                },
                            }}
                        />
                        <div className="plot-caption">
                            <h4>Caption Template:</h4>
                            <p>
                                Intra-cluster network for <strong>Cluster {selectedSubnetwork}</strong>{" "}
                                from <strong>{selectedAlgorithm}</strong>. Nodes are individual <strong>{nameColumn}</strong> values. Edges are drawn between a node and its top N nearest neighbors (default = 5 or user-selected)
                                and represent the applied proximity in 2D projection (<strong>{selectedReduction}</strong>).
                            </p>
                        </div>

                        {selectedProfilesData.length > 0 && (
                            <div style={{ marginTop: '30px' }}>
                                <Plot
                                    data={selectedProfilesData}
                                    layout={selectedProfilesLayout}
                    useResizeHandler
                    style={{ width: '100%' }}
                                    config={{
                                        responsive: true,
                                        displayModeBar: true,
                                        toImageButtonOptions: {
                                            format: "svg",
                                            filename: `selected_profiles_${selectedSubnetwork}`,
                                            scale: 1,
                                        },
                                    }}
                                />
                                <div className="plot-caption">
                                    <h4>Selected Profiles:</h4>
                                    <p>
                                        Log10 of the raw profiles for: <strong>{selectedProfileNames.join(', ')}</strong>.
                                    </p>
                                </div>
                            </div>
                        )}
                        <div style={{ textAlign: 'center', marginTop: '10px' }}>
                            <button onClick={() => setSubnetworkData([])}>Clear Subnetwork</button>
                        </div>
                    </>
                </section>
            )}
            
            </div>

            <div style={{
                backgroundColor: '#f0f8ff',
                padding: '20px',
                borderRadius: '10px',
                marginTop: '20px',
                marginBottom: '20px'
              }}>
                <h3 style={{ color: '#6a1b9a', marginBottom: '10px' }}>Method Details:<HelpTip text={HELP.methodDetails} label="method details" /></h3>
                <pre style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>{methodDetails}</pre>
                {methodsTemplateText && (
                  <p style={{ marginTop: '12px', marginBottom: '12px', lineHeight: 1.6 }}>
                    <strong>Methods Template:</strong> {methodsTemplateText}
                  </p>
                )}
                <button onClick={downloadMethodDetails} style={{
                  marginTop: '10px',
                  padding: '8px 12px',
                  backgroundColor: '#555',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}>
                  Download Method Details
                </button>
              </div>

            <div className="cluster-details">
                <div className="cluster-details-header">
                  <h3>
                    Cluster Details ({sampling?.sampled
                      ? `${sortedClusterData.length.toLocaleString()} of ${sampling.total.toLocaleString()} rows`
                      : `${sortedClusterData.length} rows`})
                    <HelpTip text={HELP.clusterDetails} label="cluster details" />
                  </h3>
                  <button
                    type="button"
                    className="details-toggle"
                    onClick={() => setShowClusterDetails(show => !show)}
                    aria-expanded={showClusterDetails}
                  >
                    {showClusterDetails ? 'Hide table ▲' : 'Show table ▼'}
                  </button>
                </div>
                <button
                  onClick={handleDownloadTSV}
                  className="download-button" // Add a class for styling
                  disabled={!clusterData || clusterData.length === 0}
                >
                  {sampling?.sampled ? `Download TSV (all ${sampling.total.toLocaleString()} rows)` : 'Download TSV'}
                </button>
                {showClusterDetails && (
                <table>
                    <thead>
                        <tr>
                            {renderSortableHeader('name', 'Name')}
                            {renderSortableHeader('cluster', 'Cluster')}
                            {renderSortableHeader('x', 'X')}
                            {renderSortableHeader('y', 'Y')}
                            {hasMembership && renderSortableHeader('membership', 'Membership')}
                            {hasRowStability && renderSortableHeader('stability', 'Stability')}
                            {renderSortableHeader('color', 'Color', { width: '70px' })}
                        </tr>
                    </thead>
                    <tbody>
                        {sortedClusterData.map((d, index) => (
                            <tr key={index}>
                                <td>{d.name}</td>
                                <td>{d.cluster === -1 ? 'Noise (-1)' : d.cluster}</td>
                                <td>{d.x.toFixed(3)}</td>
                                <td>{d.y.toFixed(3)}</td>
                                {hasMembership && <td>{formatShare(d.membership)}</td>}
                                {hasRowStability && <td>{formatShare(d.stability)}</td>}
                                <td style={{ backgroundColor: d.color, border: '1px solid #ccc' }}></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                )}
            </div>
          </section>
        )}
      </main>
      <Modal
          isOpen={showReductionModal}
          onRequestClose={() => setShowReductionModal(false)}
          contentLabel="Dimensionality Reduction Info"
          className="guide-modal"
          style={{ content: { maxWidth: '700px', margin: 'auto' } }}
        >
          <AlgorithmGuideDR />
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <button
              onClick={() => setShowReductionModal(false)}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6a1b9a',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
          </div>
        </Modal>

        <Modal
          isOpen={showClusteringModal}
          onRequestClose={() => setShowClusteringModal(false)}
          contentLabel="Clustering Algorithms Info"
          className="guide-modal"
          style={{ content: { maxWidth: '700px', margin: 'auto' } }}
        >
          <AlgorithmGuideClustering />
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <button
              onClick={() => setShowClusteringModal(false)}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6a1b9a',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer'
              }}
            >
              Close
            </button>
          </div>
        </Modal>
    </div>
  );
}

export default App;

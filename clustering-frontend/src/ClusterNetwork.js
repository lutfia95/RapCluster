import React, { useEffect, useMemo, useRef, useState } from 'react';
import cytoscape from 'cytoscape';

// Network of cluster centroids. Nodes start at the cluster centre in the 2D embedding and can be dragged;
// hover texts always show the original distances. Clicking a node opens its sub-network.

const LAYOUT_SIZE = 600; // pixels spanned by the larger axis of the embedding before fitting
const MIN_NODE = 10;
const MAX_NODE = 30;

const clusterLabel = (cluster) => (cluster === -1 ? 'Noise (-1)' : `Cluster ${cluster}`);

export function buildNetwork(clusterData, clusterColors, clusterSizes) {
  const groups = {};
  (clusterData || []).forEach(d => {
    if (!groups[d.cluster]) groups[d.cluster] = { sx: 0, sy: 0, n: 0 };
    groups[d.cluster].sx += d.x;
    groups[d.cluster].sy += d.y;
    groups[d.cluster].n += 1;
  });
  const centroids = Object.entries(groups).map(([cluster, g]) => ({
    cluster: parseInt(cluster, 10),
    x: g.sx / g.n,
    y: g.sy / g.n,
    size: clusterSizes?.[cluster] ?? g.n,
  }));
  if (!centroids.length) return { nodes: [], edges: [] };

  const xs = centroids.map(c => c.x);
  const ys = centroids.map(c => c.y);
  const range = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
  const scale = LAYOUT_SIZE / range;
  const maxSize = Math.max(...centroids.map(c => c.size));

  const nodes = centroids.map(c => ({
    id: `c${c.cluster}`,
    cluster: c.cluster,
    label: clusterLabel(c.cluster),
    size: c.size,
    color: clusterColors?.[c.cluster] || '#000000',
    diameter: maxSize ? MIN_NODE + (MAX_NODE - MIN_NODE) * (c.size / maxSize) : MIN_NODE,
    // Screen y grows downwards, so the embedding's y axis is flipped.
    position: { x: c.x * scale, y: -c.y * scale },
  }));

  const edges = [];
  for (let i = 0; i < centroids.length; i += 1) {
    for (let j = i + 1; j < centroids.length; j += 1) {
      const a = centroids[i];
      const b = centroids[j];
      edges.push({ id: `e${a.cluster}_${b.cluster}`, a, b, distance: Math.hypot(a.x - b.x, a.y - b.y) });
    }
  }
  const maxDistance = Math.max(...edges.map(e => e.distance), 0) || 1;
  return {
    nodes,
    edges: edges.map(e => {
      const normalized = e.distance / maxDistance;
      const grey = Math.floor(255 * normalized);
      return {
        id: e.id,
        source: `c${e.a.cluster}`,
        target: `c${e.b.cluster}`,
        distance: e.distance,
        label: `${clusterLabel(e.a.cluster)} ↔ ${clusterLabel(e.b.cluster)}`,
        width: Math.min(3, 1 + 2 / Math.max(e.distance, 0.01)),
        color: `rgb(${grey}, ${grey}, ${grey})`,
        opacity: Math.max(0.2, 1 - normalized * 0.8),
      };
    }),
  };
}

const escapeXml = (text) => String(text).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

// Vector export of the network as currently arranged.
function networkSvg(cy) {
  const box = cy.elements().boundingBox({ includeLabels: true });
  const pad = 20;
  const titleHeight = 30;
  const width = box.w + 2 * pad;
  const height = box.h + 2 * pad + titleHeight;
  const tx = (x) => (x - box.x1 + pad).toFixed(1);
  const ty = (y) => (y - box.y1 + pad + titleHeight).toFixed(1);
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width.toFixed(0)}" height="${height.toFixed(0)}" viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}" font-family="Arial, sans-serif">`,
    `<rect width="100%" height="100%" fill="#fcfcfc"/>`,
    `<text x="${(width / 2).toFixed(1)}" y="22" text-anchor="middle" font-size="16">Network of Cluster Centroids</text>`,
  ];
  cy.edges().forEach(edge => {
    const s = edge.source().position();
    const t = edge.target().position();
    parts.push(`<line x1="${tx(s.x)}" y1="${ty(s.y)}" x2="${tx(t.x)}" y2="${ty(t.y)}" stroke="${edge.data('color')}" stroke-opacity="${edge.data('opacity').toFixed(3)}" stroke-width="${edge.data('width').toFixed(2)}"/>`);
  });
  cy.nodes().forEach(node => {
    const p = node.position();
    const r = node.data('diameter') / 2;
    parts.push(`<circle cx="${tx(p.x)}" cy="${ty(p.y)}" r="${r.toFixed(1)}" fill="${node.data('color')}" stroke="#000" stroke-width="1"/>`);
    parts.push(`<text x="${tx(p.x)}" y="${ty(p.y - r - 4)}" text-anchor="middle" font-size="12">${escapeXml(node.data('label'))}</text>`);
  });
  parts.push('</svg>');
  return parts.join('\n');
}

function download(href, filename) {
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

// savedPositions: a ref owned by the parent, so a dragged layout survives switching plot tabs.
function ClusterNetwork({ clusterData, clusterColors, clusterSizes, savedPositions, onClusterClick }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const clickRef = useRef(onClusterClick);
  const [tooltip, setTooltip] = useState(null);
  clickRef.current = onClusterClick;

  const network = useMemo(
    () => buildNetwork(clusterData, clusterColors, clusterSizes),
    [clusterData, clusterColors, clusterSizes],
  );

  useEffect(() => {
    if (!containerRef.current || !network.nodes.length) return undefined;
    const saved = savedPositions?.current || {};
    const cy = cytoscape({
      container: containerRef.current,
      elements: [
        ...network.nodes.map(node => ({
          group: 'nodes',
          data: { ...node, position: undefined },
          position: saved[node.id] ? { ...saved[node.id] } : { ...node.position },
        })),
        ...network.edges.map(edge => ({ group: 'edges', data: edge })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            width: 'data(diameter)',
            height: 'data(diameter)',
            'background-color': 'data(color)',
            'border-width': 1,
            'border-color': '#000',
            label: 'data(label)',
            'font-size': 12,
            'text-valign': 'top',
            'text-margin-y': -4,
            color: '#222',
          },
        },
        { selector: 'node:active', style: { 'overlay-opacity': 0.1 } },
        {
          selector: 'edge',
          style: {
            width: 'data(width)',
            'line-color': 'data(color)',
            opacity: 'data(opacity)',
            'curve-style': 'straight',
          },
        },
        { selector: 'edge.hovered', style: { 'line-color': '#e67e22', opacity: 1 } },
      ],
      layout: { name: 'preset', fit: true, padding: 40 },
      // Wheel zoom would hijack page scrolling; the view is fitted instead.
      userZoomingEnabled: false,
      boxSelectionEnabled: false,
      autoungrabify: false,
    });
    cyRef.current = cy;

    const showTip = (event, text) => {
      const p = event.renderedPosition || event.target.renderedMidpoint?.() || { x: 0, y: 0 };
      setTooltip({ x: p.x, y: p.y, text });
    };
    cy.on('mouseover', 'node', e => {
      containerRef.current.style.cursor = 'grab';
      showTip(e, `${e.target.data('label')} · Candidates: ${e.target.data('size').toLocaleString()}`);
    });
    cy.on('mouseover', 'edge', e => {
      e.target.addClass('hovered');
      showTip(e, `${e.target.data('label')} · Distance: ${e.target.data('distance').toFixed(3)}`);
    });
    cy.on('mouseout', 'node, edge', e => {
      e.target.removeClass('hovered');
      containerRef.current.style.cursor = '';
      setTooltip(null);
    });
    cy.on('grab', 'node', () => setTooltip(null));
    cy.on('dragfree', 'node', e => {
      if (savedPositions) savedPositions.current = { ...savedPositions.current, [e.target.id()]: { ...e.target.position() } };
    });
    cy.on('tap', 'node', e => clickRef.current?.(e.target.data('cluster')));

    // The canvas can be hidden (width 0) or still settling when this mounts, and the panel width changes with
    // the window: refit whenever the canvas gets a new, non-zero size. Zooming is off, so nothing is lost.
    let lastSize = '';
    const observer = new ResizeObserver(() => {
      const el = containerRef.current;
      if (!el || !el.clientWidth || !el.clientHeight) return;
      const size = `${el.clientWidth}x${el.clientHeight}`;
      if (size === lastSize) return;
      lastSize = size;
      cy.resize();
      cy.fit(undefined, 40);
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, [network, savedPositions]);

  const resetLayout = () => {
    const cy = cyRef.current;
    if (!cy) return;
    if (savedPositions) savedPositions.current = {};
    network.nodes.forEach(node => cy.getElementById(node.id).position({ ...node.position }));
    cy.fit(undefined, 40);
  };

  const downloadSvg = () => {
    if (!cyRef.current) return;
    const url = URL.createObjectURL(new Blob([networkSvg(cyRef.current)], { type: 'image/svg+xml' }));
    download(url, 'network_plot.svg');
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const downloadPng = () => {
    if (!cyRef.current) return;
    download(cyRef.current.png({ full: true, scale: 3, bg: '#fcfcfc' }), 'network_plot.png');
  };

  if (!network.nodes.length) return null;

  return (
    <div className="cluster-network">
      <div className="cluster-network-header">
        <h4>Network of Cluster Centroids</h4>
        <div className="cluster-network-actions">
          <button type="button" onClick={resetLayout}>Reset layout</button>
          <button type="button" onClick={() => cyRef.current?.fit(undefined, 40)}>Fit to view</button>
          <button type="button" onClick={downloadSvg}>Download SVG</button>
          <button type="button" onClick={downloadPng}>Download PNG</button>
        </div>
      </div>
      <small className="cluster-network-hint">
        Drag a node to move it; click a cluster to open its sub-network. Hover texts always show the distances in
        the 2D embedding, also after nodes were moved.
      </small>
      {/* Cytoscape owns the canvas div; the tooltip is a sibling so React and Cytoscape never share children. */}
      <div className="cluster-network-stage">
        <div className="cluster-network-canvas" ref={containerRef} />
        {tooltip && (
          <div className="cluster-network-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
            {tooltip.text}
          </div>
        )}
      </div>
    </div>
  );
}

export default ClusterNetwork;

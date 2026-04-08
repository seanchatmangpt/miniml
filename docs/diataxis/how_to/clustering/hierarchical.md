# Hierarchical Clustering

Cluster data with agglomerative clustering and linkage methods.

## Problem

You want to cluster your data but do not know the number of clusters in advance. K-Means requires specifying k; DBSCAN requires tuning epsilon. Hierarchical clustering builds a tree of clusters (dendrogram) and lets you decide where to cut -- giving you flexibility and visual insight into the data structure.

## Solution

Use hierarchical agglomerative clustering. It starts with each point as its own cluster and merges the closest pairs until all points are in one cluster. The linkage method determines how "distance" between clusters is measured.

### Step 1: Try different linkage methods

```typescript
import { hierarchicalClustering, standardScaler, silhouetteScore } from "@seanchatmangpt/wminml";

const nSamples = X.length / nFeatures;
const { scaled } = standardScaler(X, nSamples, nFeatures);

const linkages = ["single", "average", "complete"] as const;

for (const linkage of linkages) {
  console.log(`\nLinkage: ${linkage}`);

  for (let k = 2; k <= 6; k++) {
    const { labels } = hierarchicalClustering(
      scaled,
      nSamples,
      nFeatures,
      k,
      linkage
    );
    const score = silhouetteScore(scaled, labels, nSamples, nFeatures, k);
    console.log(`  k=${k}: silhouette=${score.toFixed(4)}`);
  }
}
```

### Step 2: Pick the best linkage and k

```typescript
let bestScore = -Infinity;
let bestK = 2;
let bestLinkage = "average";
let bestLabels: Float64Array = new Float64Array();

for (const linkage of linkages) {
  for (let k = 2; k <= 8; k++) {
    const { labels } = hierarchicalClustering(
      scaled,
      nSamples,
      nFeatures,
      k,
      linkage
    );
    const score = silhouetteScore(scaled, labels, nSamples, nFeatures, k);

    if (score > bestScore) {
      bestScore = score;
      bestK = k;
      bestLinkage = linkage;
      bestLabels = labels;
    }
  }
}

console.log(`Best: linkage=${bestLinkage}, k=${bestK}, silhouette=${bestScore.toFixed(4)}`);
```

### Step 3: Compare with K-Means and DBSCAN

```typescript
import { kmeans, dbscan } from "@seanchatmangpt/wminml";

// K-Means with same k
const { labels: kmLabels } = kmeans(scaled, nSamples, nFeatures, bestK, 100);
const kmScore = silhouetteScore(scaled, kmLabels, nSamples, nFeatures, bestK);

// DBSCAN (density-based, auto-detects k)
const { labels: dbLabels, nClusters: dbK } = dbscan(
  scaled,
  nSamples,
  nFeatures,
  0.5,
  5
);
const dbScore =
  dbK > 1
    ? silhouetteScore(scaled, dbLabels, nSamples, nFeatures, dbK)
    : 0;

console.log("\nAlgorithm Comparison:");
console.log(`  Hierarchical (${bestLinkage}): ${bestScore.toFixed(4)}`);
console.log(`  K-Means (k=${bestK}):            ${kmScore.toFixed(4)}`);
console.log(`  DBSCAN (${dbK} clusters):        ${dbScore.toFixed(4)}`);
```

### Linkage methods explained

| Linkage | How It Works | Best For |
|---------|-------------|----------|
| `single` | Distance between closest points in two clusters | Elongated, chain-like clusters |
| `average` | Average distance between all point pairs | Most general-purpose, balanced |
| `complete` | Distance between farthest points in two clusters | Compact, equally-sized clusters |

### When to use hierarchical over K-Means

| Condition | Prefer |
|-----------|--------|
| You do not know k in advance | Hierarchical |
| You need a hierarchy of cluster resolutions | Hierarchical |
| Clusters are roughly spherical and similar size | K-Means (faster) |
| Dataset is large (> 10k samples) | K-Means (hierarchical is O(n^2)) |
| You want to explore different numbers of clusters | Hierarchical (compute once, cut at different levels) |

## Tips

- Always scale features first. Hierarchical clustering uses distance.
- Start with `average` linkage -- it is the most robust choice for most datasets.
- `single` linkage can produce "chaining" effects where clusters merge one point at a time.
- `complete` linkage tends to produce compact, equally-sized clusters.
- For large datasets (> 1000 samples), hierarchical clustering becomes slow. Use K-Means or DBSCAN instead.

## See Also

- [Choose K for K-Means](choose-k.md) -- when you have spherical clusters
- [Handle Arbitrary Shapes](dbscan.md) -- when clusters are irregular
- [Scale Your Features](../preprocessing/scaling.md) -- required before clustering

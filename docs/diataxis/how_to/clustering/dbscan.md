# Handle Arbitrary Shapes

Cluster data with DBSCAN when clusters are not spherical.

## Problem

K-Means assumes clusters are convex and roughly spherical. Real-world data often has elongated, crescent-shaped, or irregularly distributed clusters. K-Means will incorrectly split or merge these shapes. You need an algorithm that finds clusters based on density, not distance to a centroid.

## Solution

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are closely packed together and marks points in low-density regions as noise. It does not require specifying the number of clusters.

### Step 1: Choose epsilon and minPoints

`epsilon` defines the neighborhood radius. `minPoints` is the minimum number of points within that radius to form a dense region.

```typescript
import { standardScaler, dbscan, silhouetteScore } from "@seanchatmangpt/wminml";

const nSamples = X.length / nFeatures;
const { scaled } = standardScaler(X, nSamples, nFeatures);

// Try a range of epsilon values
const epsilons = [0.3, 0.5, 0.7, 1.0, 1.5];
const minPts = Math.max(3, Math.floor(nFeatures * 2)); // rule of thumb

console.log("Epsilon | Clusters | Noise Points | Silhouette");
console.log("--------|----------|--------------|-----------");

for (const eps of epsilons) {
  const { labels, nClusters } = dbscan(scaled, nSamples, nFeatures, eps, minPts);

  // Count noise points (labeled -1 in DBSCAN)
  const noiseCount = Array.from(labels).filter((l) => l < 0).length;

  let score = 0;
  if (nClusters > 1 && nClusters < nSamples - 1) {
    score = silhouetteScore(scaled, labels, nSamples, nFeatures, nClusters);
  }

  console.log(
    `${eps.toFixed(1).padStart(7)} | ${(nClusters + "").padStart(8)} | ${(noiseCount + "").padStart(12)} | ${score.toFixed(4)}`
  );
}
```

### Step 2: Apply DBSCAN with the best parameters

```typescript
const { labels, nClusters } = dbscan(scaled, nSamples, nFeatures, 0.5, 5);

console.log(`Found ${nClusters} clusters`);
console.log(`Noise points: ${Array.from(labels).filter((l) => l < 0).length}`);

// Inspect cluster membership
const clusterCounts = new Map<number, number>();
for (const label of labels) {
  clusterCounts.set(label, (clusterCounts.get(label) || 0) + 1);
}

for (const [cluster, count] of clusterCounts) {
  const name = cluster < 0 ? "Noise" : `Cluster ${cluster}`;
  console.log(`  ${name}: ${count} samples`);
}
```

### Step 3: Compare with K-Means

```typescript
import { kmeans } from "@seanchatmangpt/wminml";

// Run K-Means with k matching DBSCAN cluster count
const { labels: kmLabels } = kmeans(scaled, nSamples, nFeatures, nClusters, 100);
const kmScore = silhouetteScore(scaled, kmLabels, nSamples, nFeatures, nClusters);

const dbScore = silhouetteScore(scaled, labels, nSamples, nFeatures, nClusters);

console.log(`\nK-Means (k=${nClusters}): silhouette=${kmScore.toFixed(4)}`);
console.log(`DBSCAN:                 silhouette=${dbScore.toFixed(4)}`);

// DBSCAN typically wins on non-spherical data
```

### Choosing epsilon

| Guideline | Value |
|-----------|-------|
| Start with | `epsilon = 0.5` (on scaled data) |
| Too small (< 0.1) | Everything becomes noise |
| Too large (> 2.0) | Everything becomes one cluster |
| Good range | Most points clustered, few noise points |

### Choosing minPoints

| Guideline | Value |
|-----------|-------|
| Minimum | `nFeatures + 1` |
| Default | `2 * nFeatures` |
| Larger dataset | `2 * nFeatures` or higher |
| Noisy data | Higher (more resistant to noise) |

## Tips

- Always scale features first. DBSCAN is distance-based.
- If DBSCAN finds only 1 cluster, decrease epsilon. If everything is noise, increase epsilon.
- Points labeled -1 are noise -- they do not belong to any cluster. This is a feature, not a bug.
- DBSCAN cannot handle clusters of vastly different densities. The same epsilon applies to all clusters.
- If silhouette score is low for both K-Means and DBSCAN, the data may not have meaningful clusters.

## See Also

- [Choose K for K-Means](choose-k.md) -- when clusters are roughly spherical
- [Hierarchical Clustering](hierarchical.md) -- alternative density-based approach
- [Scale Your Features](../preprocessing/scaling.md) -- required before DBSCAN

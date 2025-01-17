from phy import IPlugin, connect
import logging
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
import umap

logger = logging.getLogger('phy')


class ReclusterUMAP(IPlugin):
    """
    Modern spike sorting plugin with optimized performance and intelligent merging
    """

    def __init__(self):
        super(ReclusterUMAP, self).__init__()
        self._shortcuts_created = False
        self._umap_reducer = None
        self._last_n_spikes = None

    def attach_to_controller(self, controller):
        def prepareFeatures(spikeIds):
            data = controller.model._load_features().data[spikeIds]
            features = np.reshape(data, (data.shape[0], -1))
            return features

        def compute_template_correlation(features, labels):
            """Compute correlation between cluster templates"""
            n_clusters = len(np.unique(labels[labels > 0]))
            correlations = np.zeros((n_clusters, n_clusters))

            for i in range(n_clusters):
                template_i = np.mean(features[labels == i + 1], axis=0)
                for j in range(i + 1, n_clusters):
                    template_j = np.mean(features[labels == j + 1], axis=0)
                    # Correlation between templates
                    corr = np.corrcoef(template_i, template_j)[0, 1]
                    correlations[i, j] = correlations[j, i] = corr

            return correlations

        def check_spatial_consistency(features, labels, threshold=0.6):
            """Check if clusters are spatially consistent"""
            n_clusters = len(np.unique(labels[labels > 0]))
            spatial_consistent = np.zeros((n_clusters, n_clusters), dtype=bool)

            # Assuming features contain channel information
            for i in range(n_clusters):
                spikes_i = features[labels == i + 1]
                channels_i = np.var(spikes_i, axis=0).argsort()[-4:]  # Top 4 channels

                for j in range(i + 1, n_clusters):
                    spikes_j = features[labels == j + 1]
                    channels_j = np.var(spikes_j, axis=0).argsort()[-4:]

                    # Check channel overlap
                    common_channels = len(set(channels_i) & set(channels_j))
                    spatial_consistent[i, j] = spatial_consistent[j, i] = (
                            common_channels >= len(channels_i) * threshold
                    )

            return spatial_consistent

        def merge_similar_clusters(features, labels, template_threshold=0.9, spatial_threshold=0.6):
            """Merge clusters based on template similarity and spatial consistency"""
            while True:
                n_clusters = len(np.unique(labels[labels > 0]))
                if n_clusters <= 2:  # Don't merge if only 2 clusters remain
                    break

                # Compute similarity matrices
                correlations = compute_template_correlation(features, labels)
                spatial_consistent = check_spatial_consistency(features, labels, spatial_threshold)

                # Find most similar pair that's spatially consistent
                max_corr = template_threshold
                merge_pair = None

                for i in range(n_clusters):
                    for j in range(i + 1, n_clusters):
                        if (correlations[i, j] > max_corr and spatial_consistent[i, j]):
                            max_corr = correlations[i, j]
                            merge_pair = (i, j)

                if merge_pair is None:
                    break

                # Perform merge
                i, j = merge_pair
                new_labels = np.zeros_like(labels)
                for k in range(n_clusters):
                    if k == i:
                        new_labels[labels == k + 1] = i + 1
                    elif k == j:
                        new_labels[labels == j + 1] = i + 1
                    elif k > j:
                        new_labels[labels == k + 1] = k
                    else:
                        new_labels[labels == k + 1] = k + 1

                labels = new_labels
                logger.info(f"Merged clusters {i + 1} and {j + 1} (correlation: {max_corr:.3f})")

            return labels

        def fastClustering(embedding, target_clusters=4):
            """Fast clustering with intelligent merging"""
            # Initial over-clustering
            initial_clusters = min(target_clusters * 3, len(embedding) // 50)

            # Initial clustering
            kmeans = MiniBatchKMeans(
                n_clusters=initial_clusters,
                batch_size=1000,
                random_state=42
            )
            initial_labels = kmeans.fit_predict(embedding) + 1  # Make labels 1-based

            # Merge similar clusters
            final_labels = merge_similar_clusters(
                embedding,
                initial_labels,
                template_threshold=0.9,
                spatial_threshold=0.6
            )

            return final_labels

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+k', prompt=True, prompt_default=lambda: 4)
            def umapGmmClustering(target_clusters):
                """Fast UMAP-GMM Clustering with intelligent merging (Alt+K)"""
                try:
                    target_clusters = int(target_clusters)
                    if target_clusters < 2:
                        logger.warn("Need at least 2 clusters, using 2")
                        target_clusters = 2

                    clusterIds = controller.supervisor.selected
                    if not clusterIds:
                        logger.warn("No clusters selected!")
                        return

                    bunchs = controller._amplitude_getter(clusterIds, name='template', load_all=True)
                    spikeIds = bunchs[0].spike_ids
                    n_spikes = len(spikeIds)
                    logger.info(f"Processing {n_spikes} spikes with target {target_clusters} clusters")

                    # Feature preparation
                    features = prepareFeatures(spikeIds)
                    scaler = StandardScaler()
                    featuresScaled = scaler.fit_transform(features)

                    # Dimensionality reduction
                    pca = PCA(n_components=min(30, featuresScaled.shape[1]))
                    featuresPca = pca.fit_transform(featuresScaled)

                    # UMAP reduction
                    if (self._umap_reducer is None or
                            self._last_n_spikes is None or
                            abs(self._last_n_spikes - n_spikes) > n_spikes * 0.2):
                        self._umap_reducer = umap.UMAP(
                            n_neighbors=min(30, n_spikes // 100),
                            min_dist=0.2,
                            n_components=2,
                            random_state=42,
                            n_jobs=-1,
                            metric='euclidean',
                            low_memory=True
                        )
                        self._last_n_spikes = n_spikes

                    embedding = self._umap_reducer.fit_transform(featuresPca)

                    # Clustering with merging
                    labels = fastClustering(embedding, target_clusters)
                    n_clusters = len(np.unique(labels))

                    logger.info(f"Created {n_clusters} clusters after merging")
                    controller.supervisor.actions.split(spikeIds, labels)

                except Exception as e:
                    logger.error(f"Error in umapGmmClustering: {str(e)}")

            # Keep the existing templateBasedSplit function unchanged
            @controller.supervisor.actions.add(shortcut='alt+t', prompt=True, prompt_default=lambda: 0.85)
            def templateBasedSplit(similarityThreshold):
                """Template-based spike sorting (Alt+T)"""
                # ... [rest of the existing templateBasedSplit code remains unchanged]
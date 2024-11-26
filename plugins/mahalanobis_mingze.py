from phy import IPlugin, connect
import logging
import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger('phy')


class StableMahalanobisDetectionMingze(IPlugin):
    """Numerically stable Mahalanobis-based spike sorting plugin"""

    def __init__(self):
        super(StableMahalanobisDetectionMingze, self).__init__()
        self._shortcuts_created = False

    def attach_to_controller(self, controller):
        def prepare_features(spike_ids):
            """Prepare feature matrix from spike data"""
            data = controller.model._load_features().data[spike_ids]
            return np.reshape(data, (data.shape[0], -1))

        def stable_mahalanobis(X):
            """
            Compute Mahalanobis distances with numerical stability safeguards
            """
            # Initial standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Compute covariance matrix with regularization
            cov = np.cov(X_scaled, rowvar=False)
            n_features = cov.shape[0]

            # Add small regularization to prevent singular matrices
            cov += np.eye(n_features) * 1e-6

            try:
                # Use SVD for numerical stability
                U, s, Vt = np.linalg.svd(cov)

                # Filter small singular values
                s[s < 1e-10] = 1e-10

                # Reconstruct inverse covariance
                inv_cov = (U / s) @ Vt

                # Compute mean
                mu = np.mean(X_scaled, axis=0)

                # Compute distances
                diff = X_scaled - mu
                distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

                return distances

            except np.linalg.LinAlgError:
                logger.error("SVD failed, falling back to diagonal covariance")
                # Fallback: use only diagonal elements
                inv_cov = np.diag(1.0 / np.diag(cov))
                mu = np.mean(X_scaled, axis=0)
                diff = X_scaled - mu
                return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+x', prompt=True, prompt_default=lambda: '14')
            def stable_mahalanobis_outliers(threshold_std):
                """
                Stable Mahalanobis Outlier Detection (Alt+X)
                Enter threshold in standard deviations (e.g., 14)
                """
                try:
                    # Parse threshold
                    threshold_std = float(threshold_std)
                    if threshold_std <= 0:
                        logger.warn("Threshold must be positive, using 14")
                        threshold_std = 14

                    # Get selected clusters and spikes
                    cluster_ids = controller.supervisor.selected
                    if not cluster_ids:
                        logger.warn("No clusters selected!")
                        return

                    bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                    spike_ids = bunchs[0].spike_ids

                    # Prepare features
                    features = prepare_features(spike_ids)

                    # Minimum spikes check
                    if features.shape[0] < features.shape[1] * 2:
                        logger.warn(f"Warning: Need at least {features.shape[1] * 2} spikes!")
                        return

                    # Compute distances
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        distances = stable_mahalanobis(features)

                    # Threshold based on chi-square distribution
                    dof = features.shape[1]
                    chi_threshold = chi2.ppf(0.999, dof)  # Base threshold
                    scale_factor = (threshold_std ** 2) / chi_threshold

                    # Apply threshold
                    outliers = distances > np.sqrt(chi_threshold * scale_factor)

                    # Prepare labels
                    labels = np.ones(len(spike_ids), dtype=int)
                    labels[outliers] = 2

                    # Count outliers
                    n_outliers = np.sum(outliers)

                    # Log results
                    logger.info(f"Analysis at {threshold_std} standard deviations:")
                    logger.info(f"- Detected {n_outliers} outliers ({n_outliers / len(spike_ids) * 100:.1f}%)")
                    logger.info(f"- Maximum distance: {np.max(distances):.1f}")
                    logger.info(f"- 95th percentile: {np.percentile(distances, 95):.1f}")
                    logger.info(f"- Median distance: {np.median(distances):.1f}")
                    logger.info(f"- Threshold used: {np.sqrt(chi_threshold * scale_factor):.1f}")

                    # Sort and display top distances
                    sorted_dist = np.sort(distances)[-10:]
                    logger.info(f"Top 10 distances: {', '.join(f'{d:.1f}' for d in sorted_dist)}")

                    # Split if outliers found
                    if n_outliers > 0:
                        controller.supervisor.actions.split(spike_ids, labels)
                    else:
                        logger.info("No outliers detected at current threshold")

                except Exception as e:
                    logger.error(f"Error in stable_mahalanobis_outliers: {str(e)}")
                    logger.error(f"Stack trace:", exc_info=True)
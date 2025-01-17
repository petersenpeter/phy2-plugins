from phy import IPlugin, connect
import logging
import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtWidgets, QtCore
import seaborn as sns

logger = logging.getLogger('phy')


class StableMahalanobisDetection(IPlugin):
    def __init__(self):
        super(StableMahalanobisDetection, self).__init__()
        self._shortcuts_created = False
        self.current_distances = None
        self.current_threshold = None
        self.plot_window = None
        self._spike_ids = None
        self._n_features = None
        self._feature_structure = None

    def attach_to_controller(self, controller):
        def get_feature_dimensions(features_arr):
            """Analyze the feature array structure to get actual dimensions"""
            try:
                # Get the feature dimensions from the model
                feature_shape = controller.model._load_features().data.shape
                if len(feature_shape) == 3:  # (n_spikes, n_channels, n_pcs)
                    n_channels = feature_shape[1]
                    n_pcs = feature_shape[2]
                    logger.info(f"Feature structure: {n_channels} channels with {n_pcs} PCs each")
                    return n_channels * n_pcs
                else:
                    logger.warn(f"Unexpected feature shape: {feature_shape}")
                    return features_arr.shape[1]
            except Exception as e:
                logger.error(f"Error getting feature dimensions: {str(e)}")
                return features_arr.shape[1]

        def prepare_features(spike_ids):
            """Prepare feature matrix from spike data with proper dimensionality"""
            try:
                # Load features with original structure
                features = controller.model._load_features().data[spike_ids]

                # Log feature shape information
                logger.info(f"Original feature shape: {features.shape}")

                # Reshape maintaining actual structure
                features_flat = np.reshape(features, (features.shape[0], -1))

                # Get actual feature dimensions
                self._n_features = get_feature_dimensions(features)
                logger.info(f"Using {self._n_features} feature dimensions for Mahalanobis distance")

                return features_flat

            except Exception as e:
                logger.error(f"Error preparing features: {str(e)}")
                return None

        def stable_mahalanobis(X):
            """Compute Mahalanobis distances with numerical stability safeguards"""
            if X is None or len(X) == 0:
                logger.error("Empty or invalid feature matrix")
                return None

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                cov = np.cov(X_scaled, rowvar=False)
                n_features = cov.shape[0]
                cov += np.eye(n_features) * 1e-6

                try:
                    U, s, Vt = np.linalg.svd(cov)
                    s[s < 1e-8] = 1e-8
                    inv_cov = (U / s) @ Vt
                    mu = np.mean(X_scaled, axis=0)
                    diff = X_scaled - mu
                    distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    return np.nan_to_num(distances, nan=np.inf)

                except np.linalg.LinAlgError as e:
                    logger.error(f"SVD failed: {str(e)}, falling back to diagonal covariance")
                    inv_cov = np.diag(1.0 / np.diag(cov))
                    mu = np.mean(X_scaled, axis=0)
                    diff = X_scaled - mu
                    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

            except Exception as e:
                logger.error(f"Error in Mahalanobis distance calculation: {str(e)}")
                return None

        def calculate_robust_threshold(distances):
            """Calculate default threshold based on chi-square distribution"""
            if distances is None or len(distances) == 0 or self._n_features is None:
                return None
            # Use 99.99% chi-square threshold as default (very conservative)
            return np.sqrt(chi2.ppf(0.9999, self._n_features))

        def suggest_thresholds(distances):
            """Suggest thresholds with focus on empirical distribution"""
            if distances is None or len(distances) == 0:
                return {}

            try:
                # Calculate empirical thresholds
                empirical_thresholds = {
                    'pct_99': np.percentile(distances, 99),
                    'pct_999': np.percentile(distances, 99.9),
                    'pct_9999': np.percentile(distances, 99.99)  # More conservative
                }

                # Add chi-square thresholds if dimensionality is available
                if self._n_features is not None:
                    p = self._n_features
                    chi2_thresh_999 = np.sqrt(chi2.ppf(0.999, p))  # More conservative (0.1% false positive rate)
                    chi2_thresh_9999 = np.sqrt(chi2.ppf(0.9999, p))  # Very conservative (0.01% false positive rate)
                    empirical_thresholds['χ²_999'] = chi2_thresh_999
                    empirical_thresholds['χ²_9999'] = chi2_thresh_9999

                return empirical_thresholds

            except Exception as e:
                logger.error(f"Error calculating threshold suggestions: {str(e)}")
                return {}

        def plot_distribution(distances, threshold=None):
            """Create distribution plot with optional theoretical comparison"""
            if distances is None or len(distances) == 0:
                logger.error("No valid distances to plot")
                return

            if self.plot_window is None:
                self.plot_window = QtWidgets.QMainWindow()
                self.plot_window.setWindowTitle('Mahalanobis Distance Distribution')

            # Create widgets and layout
            widget = QtWidgets.QWidget()
            self.plot_window.setCentralWidget(widget)
            layout = QtWidgets.QVBoxLayout(widget)

            # Create figure
            fig = Figure(figsize=(12, 7))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            fig.subplots_adjust(right=0.85, bottom=0.15)

            # Plot range
            max_dist = np.max(distances)
            q99_9 = np.percentile(distances, 99.9)
            plot_max = min(max_dist, q99_9 * 1.2)

            # Plot empirical distribution
            n_bins = min(100, int(np.sqrt(len(distances))))
            sns.histplot(distances, ax=ax, bins=n_bins, stat='density')
            ax.set_xlim(0, plot_max)

            # Add theoretical comparison if dimensions are known
            if self._n_features is not None:
                x = np.linspace(0, plot_max, 200)
                chi_density = 2 * x * chi2.pdf(x ** 2, self._n_features)
                ax.plot(x, chi_density, 'r--', alpha=0.3,
                        label=f'χ² ({self._n_features} df)')

            # Labels and formatting
            ax.set_xlabel('Mahalanobis Distance', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.tick_params(labelsize=9)

            # Plot thresholds
            suggestions = suggest_thresholds(distances)
            colors = ['r', 'g', 'b', 'purple']
            for (name, value), color in zip(suggestions.items(), colors):
                if value <= plot_max:
                    n_spikes = np.sum(distances > value)
                    ax.axvline(x=value, color=color, linestyle='--',
                               label=f'{name}: {value:.1f}\n({n_spikes} spikes, {n_spikes / len(distances) * 100:.1f}%)')

            if threshold is not None and threshold <= plot_max:
                n_spikes = np.sum(distances > threshold)
                ax.axvline(x=threshold, color='black', linestyle='-',
                           label=f'Current: {threshold:.1f}\n({n_spikes} spikes, {n_spikes / len(distances) * 100:.1f}%)')

            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

            # Create input widgets
            form_layout = QtWidgets.QFormLayout()
            threshold_input = QtWidgets.QLineEdit()
            threshold_input.setPlaceholderText('Enter threshold value')
            if threshold is not None:
                threshold_input.setText(str(threshold))

            def on_return_pressed():
                apply_button.click()

            threshold_input.returnPressed.connect(on_return_pressed)

            form_layout.addRow("Threshold:", threshold_input)

            # Button layout
            button_layout = QtWidgets.QHBoxLayout()
            button_layout.setSpacing(10)

            # Add preset buttons
            for name, value in suggestions.items():
                preset_button = QtWidgets.QPushButton(f'Use {name}')
                preset_button.setMinimumWidth(100)
                preset_button.clicked.connect(lambda checked, v=value: threshold_input.setText(f"{v:.2f}"))
                button_layout.addWidget(preset_button)

            # Preview and Apply buttons
            preview_button = QtWidgets.QPushButton('Preview Selection')
            apply_button = QtWidgets.QPushButton('Apply Threshold')
            preview_button.setMinimumWidth(120)
            apply_button.setMinimumWidth(120)

            def on_preview():
                try:
                    new_threshold = float(threshold_input.text())
                    n_outliers = np.sum(distances > new_threshold)
                    QtWidgets.QMessageBox.information(
                        self.plot_window, 'Preview',
                        f'This threshold would mark {n_outliers} spikes ({n_outliers / len(distances) * 100:.2f}%) as outliers.\n'
                        f'Maximum distance: {np.max(distances):.1f}\n'
                        f'99.9th percentile: {np.percentile(distances, 99.9):.1f}\n'
                        f'99th percentile: {np.percentile(distances, 99):.1f}'
                    )
                except ValueError:
                    logger.error("Invalid threshold value")

            def on_apply():
                try:
                    new_threshold = float(threshold_input.text())
                    if not new_threshold > 0:
                        logger.error("Threshold must be positive")
                        return

                    self.current_threshold = new_threshold
                    n_outliers = np.sum(distances > new_threshold)

                    if n_outliers > len(distances) * 0.1:
                        reply = QtWidgets.QMessageBox.question(
                            self.plot_window, 'Warning',
                            f'This threshold would mark {n_outliers} spikes ({n_outliers / len(distances) * 100:.1f}%) as outliers. Continue?',
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                        )
                        if reply == QtWidgets.QMessageBox.No:
                            return

                    perform_outlier_detection(new_threshold, self.current_distances)
                    self.plot_window.close()

                except ValueError:
                    logger.error("Invalid threshold value")
                except Exception as e:
                    logger.error(f"Error applying threshold: {str(e)}")

            preview_button.clicked.connect(on_preview)
            apply_button.clicked.connect(on_apply)

            # Add widgets to layout
            layout.addWidget(canvas)
            layout.addLayout(form_layout)
            layout.addLayout(button_layout)
            layout.addSpacing(10)
            button_row = QtWidgets.QHBoxLayout()
            button_row.addWidget(preview_button)
            button_row.addWidget(apply_button)
            layout.addLayout(button_row)

            # Set window size and prepare to show
            self.plot_window.resize(1600, 900)

            # Create timer to select text after window is shown
            def select_text():
                threshold_input.setFocus()
                threshold_input.selectAll()

            # Use QTimer to ensure window is fully shown
            timer = QtCore.QTimer()
            timer.singleShot(100, select_text)

            # Show window
            self.plot_window.show()

        def perform_outlier_detection(threshold, distances):
            """Perform outlier detection with given threshold"""
            if distances is None or self._spike_ids is None:
                logger.warn("No distances or spike IDs available")
                return

            try:
                outliers = distances > threshold
                n_outliers = np.sum(outliers)

                # Log results
                logger.info(f"Analysis with threshold {threshold}:")
                logger.info(f"- Detected {n_outliers} outliers ({n_outliers / len(distances) * 100:.1f}%)")
                logger.info(f"- Maximum distance: {np.max(distances):.1f}")
                logger.info(f"- 99.9th percentile: {np.percentile(distances, 99.9):.1f}")
                logger.info(f"- 99th percentile: {np.percentile(distances, 99):.1f}")
                logger.info(f"- Median distance: {np.median(distances):.1f}")

                # Sort and display top distances
                sorted_dist = np.sort(distances)[-10:]
                logger.info(f"Top 10 distances: {', '.join(f'{d:.1f}' for d in sorted_dist)}")

                # Prepare for split
                if n_outliers > 0:
                    labels = np.ones(len(distances), dtype=int)
                    labels[outliers] = 2
                    controller.supervisor.actions.split(self._spike_ids, labels)
                else:
                    logger.info("No outliers detected at current threshold")

            except Exception as e:
                logger.error(f"Error in outlier detection: {str(e)}")

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+x')
            def stable_mahalanobis_outliers():
                """
                Stable Mahalanobis Outlier Detection with visualization (Alt+X)
                """
                try:
                    # Get selected clusters and spikes
                    cluster_ids = controller.supervisor.selected
                    if not cluster_ids:
                        logger.warn("No clusters selected!")
                        return

                    bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                    self._spike_ids = bunchs[0].spike_ids

                    # Prepare features
                    features = prepare_features(self._spike_ids)
                    if features is None:
                        return

                    # Minimum spikes check
                    if features.shape[0] < features.shape[1] * 2:
                        logger.warn(f"Warning: Need at least {features.shape[1] * 2} spikes!")
                        return

                    # Compute distances
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        distances = stable_mahalanobis(features)

                    if distances is None:
                        return

                    # Store current distances
                    self.current_distances = distances

                    # Calculate initial threshold using chi-square distribution
                    initial_threshold = calculate_robust_threshold(distances)
                    if initial_threshold is None:
                        return

                    # Log distribution analysis
                    logger.info("\nDistribution Analysis:")
                    logger.info(f"Number of dimensions: {self._n_features}")
                    logger.info(f"Expected mean distance (sqrt(p)): {np.sqrt(self._n_features):.2f}")
                    logger.info(f"Observed mean distance: {np.mean(distances):.2f}")
                    logger.info(f"Observed median distance: {np.median(distances):.2f}")

                    # Check for substantial deviation from theoretical expectation
                    expected_mean = np.sqrt(self._n_features)
                    observed_mean = np.mean(distances)
                    if abs(observed_mean - expected_mean) / expected_mean > 0.5:
                        logger.warn(f"Substantial deviation from theoretical expectation:")
                        logger.warn(f"This might indicate non-normal features or other irregularities.")

                    # Show distribution plot
                    plot_distribution(distances, initial_threshold)

                except Exception as e:
                    logger.error(f"Error in stable_mahalanobis_outliers: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
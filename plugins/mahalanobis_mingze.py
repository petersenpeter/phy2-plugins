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


class StableMahalanobisDetectionMingze(IPlugin):
    """Numerically stable Mahalanobis-based spike sorting plugin with distribution visualization"""

    def __init__(self):
        super(StableMahalanobisDetectionMingze, self).__init__()
        self._shortcuts_created = False
        self.current_distances = None
        self.current_threshold = None
        self.plot_window = None
        self._spike_ids = None  # Store spike_ids as instance variable

    def attach_to_controller(self, controller):
        def prepare_features(spike_ids):
            """Prepare feature matrix from spike data"""
            try:
                data = controller.model._load_features().data[spike_ids]
                return np.reshape(data, (data.shape[0], -1))
            except Exception as e:
                logger.error(f"Error preparing features: {str(e)}")
                return None

        def stable_mahalanobis(X):
            """
            Compute Mahalanobis distances with numerical stability safeguards
            """
            if X is None or len(X) == 0:
                logger.error("Empty or invalid feature matrix")
                return None

            try:
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

                    # More aggressive filtering of small singular values
                    s[s < 1e-8] = 1e-8

                    # Reconstruct inverse covariance
                    inv_cov = (U / s) @ Vt

                    # Compute mean
                    mu = np.mean(X_scaled, axis=0)

                    # Compute distances
                    diff = X_scaled - mu
                    distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

                    # Handle any remaining numerical instabilities
                    distances = np.nan_to_num(distances, nan=np.inf)

                    return distances

                except np.linalg.LinAlgError as e:
                    logger.error(f"SVD failed: {str(e)}, falling back to diagonal covariance")
                    # Fallback: use only diagonal elements
                    inv_cov = np.diag(1.0 / np.diag(cov))
                    mu = np.mean(X_scaled, axis=0)
                    diff = X_scaled - mu
                    distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    return distances

            except Exception as e:
                logger.error(f"Error in Mahalanobis distance calculation: {str(e)}")
                return None

        def calculate_robust_threshold(distances):
            """Calculate threshold based on robust statistics"""
            if distances is None or len(distances) == 0:
                return None

            median = np.median(distances)
            mad = np.median(np.abs(distances - median))
            return median + 5 * mad

        def suggest_thresholds(distances):
            """Suggest multiple thresholds based on different methods"""
            if distances is None or len(distances) == 0:
                return {}

            try:
                median = np.median(distances)
                mad = np.median(np.abs(distances - median))
                q75 = np.percentile(distances, 75)
                iqr = np.percentile(distances, 75) - np.percentile(distances, 25)

                suggestions = {
                    'mad_5': median + 5 * mad,  # Conservative
                    'mad_7': median + 7 * mad,  # More conservative
                    'iqr_3': q75 + 3 * iqr,  # Based on IQR
                    'percentile_999': np.percentile(distances, 99.9)  # Top 0.1%
                }
                return suggestions
            except Exception as e:
                logger.error(f"Error calculating threshold suggestions: {str(e)}")
                return {}

        def plot_distribution(distances, threshold=None):
            """Create distribution plot window with interactive threshold selection"""
            if distances is None or len(distances) == 0:
                logger.error("No valid distances to plot")
                return

            if self.plot_window is None:
                self.plot_window = QtWidgets.QMainWindow()
                self.plot_window.setWindowTitle('Mahalanobis Distance Distribution')

            # Clear existing widgets
            widget = QtWidgets.QWidget()
            self.plot_window.setCentralWidget(widget)
            layout = QtWidgets.QVBoxLayout(widget)

            # Create figure with adjusted size
            fig = Figure(figsize=(12, 7))
            canvas = FigureCanvas(fig)

            # Create main plot with adjusted margins
            ax = fig.add_subplot(111)
            fig.subplots_adjust(right=0.85, bottom=0.15)

            # Calculate plot range
            max_dist = np.max(distances)
            q99_9 = np.percentile(distances, 99.9)
            plot_max = min(max_dist, q99_9 * 1.2)

            # Plot distribution
            n_bins = min(100, int(np.sqrt(len(distances))))
            sns.histplot(distances, ax=ax, bins=n_bins, stat='density')
            ax.set_yscale('log')
            ax.set_xlim(0, plot_max)

            # Labels
            ax.set_xlabel('Mahalanobis Distance', fontsize=10)
            ax.set_ylabel('Log Density', fontsize=10)
            ax.tick_params(labelsize=9)

            # Calculate and plot suggested thresholds
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

            # Add enter key support
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

                    # Warning if too many outliers
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

            # Set window size and show
            self.plot_window.resize(1600, 900)
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

                    # Calculate initial threshold
                    initial_threshold = calculate_robust_threshold(distances)
                    if initial_threshold is None:
                        return

                    # Show distribution plot
                    plot_distribution(distances, initial_threshold)

                except Exception as e:
                    logger.error(f"Error in stable_mahalanobis_outliers: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
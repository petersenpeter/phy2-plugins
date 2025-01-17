from phy import IPlugin, connect
import numpy as np
import logging
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

logger = logging.getLogger('phy')


class ImprovedISIAnalysis(IPlugin):
    """More reliable spike analysis using combined metrics"""

    def __init__(self):
        super(ImprovedISIAnalysis, self).__init__()
        self._shortcuts_created = False

    def attach_to_controller(self, controller):
        def get_waveform_features(spike_ids):
            """Extract key waveform features"""
            # Get waveforms
            data = controller.model._load_features().data[spike_ids]
            return np.reshape(data, (data.shape[0], -1))

        def analyze_suspicious_spikes(spike_times, spike_amps, waveforms, isi_threshold=0.0015):
            """
            Analyze spikes with multiple metrics:
            - ISI violations
            - Amplitude changes
            - Waveform changes
            """
            n_spikes = len(spike_times)
            suspicious = np.zeros(n_spikes, dtype=bool)

            # Find ISI violations
            isi_prev = np.diff(spike_times, prepend=spike_times[0] - 1)
            isi_next = np.diff(spike_times, append=spike_times[-1] + 1)

            # Look for changes in nearby spikes
            for i in range(n_spikes):
                if isi_prev[i] < isi_threshold or isi_next[i] < isi_threshold:
                    # For spikes with short ISI, check for:

                    # 1. Amplitude changes
                    amp_window = slice(max(0, i - 1), min(n_spikes, i + 2))
                    amp_variation = np.std(spike_amps[amp_window])

                    # 2. Waveform changes
                    wave_window = slice(max(0, i - 1), min(n_spikes, i + 2))
                    waves = waveforms[wave_window]
                    wave_distances = cdist(waves, waves, metric='correlation')
                    wave_variation = np.mean(wave_distances)

                    # Mark as suspicious if there are significant changes
                    if (amp_variation > np.std(spike_amps) * 1.5 or
                            wave_variation > 0.1):  # Correlation distance threshold
                        suspicious[i] = True

            return suspicious

        @connect
        def on_gui_ready(sender, gui):
            if self._shortcuts_created:
                return
            self._shortcuts_created = True

            @controller.supervisor.actions.add(shortcut='alt+i')
            def analyze_spike_patterns():
                """
                Analyze spike patterns using multiple metrics:
                - ISI violations
                - Amplitude changes
                - Waveform changes
                Only splits when multiple criteria suggest different units.
                """
                try:
                    # Get selected clusters
                    cluster_ids = controller.supervisor.selected
                    if not cluster_ids:
                        logger.warn("No clusters selected!")
                        return

                    # Get spike data
                    bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                    spike_ids = bunchs[0].spike_ids
                    spike_times = controller.model.spike_times[spike_ids]

                    # Get spike amplitudes
                    spike_amps = bunchs[0].amplitudes

                    # Get waveform features
                    waveforms = get_waveform_features(spike_ids)

                    # Analyze spikes
                    suspicious = analyze_suspicious_spikes(
                        spike_times,
                        spike_amps,
                        waveforms
                    )

                    # Prepare labels
                    labels = np.ones(len(spike_ids), dtype=int)
                    labels[suspicious] = 2

                    # Count suspicious spikes
                    n_suspicious = np.sum(suspicious)

                    if n_suspicious > 0:
                        # Log analysis results
                        logger.info(f"Found {n_suspicious} suspicious spikes "
                                    f"({n_suspicious / len(spike_ids) * 100:.1f}%) "
                                    f"with notable physical changes")

                        # Only split if we found enough suspicious spikes
                        if n_suspicious >= 10 and n_suspicious <= len(spike_ids) * 0.5:
                            controller.supervisor.actions.split(spike_ids, labels)
                            logger.info("Split suspicious spikes for manual review")
                        else:
                            logger.info("Too few or too many suspicious spikes for reliable splitting")
                    else:
                        logger.info("No suspicious spikes found")

                except Exception as e:
                    logger.error(f"Error in analyze_spike_patterns: {str(e)}")
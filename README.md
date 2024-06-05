# ECG-ART
# Vital Signals Analysis Dashboard

This dashboard provides an analysis of vital signals including ECG and ART data. Use the navigation menu to explore different sections of the analysis.

## Setup Instructions

1. **Download the Vital File**:
   - Visit [PhysioNet VitalDB](https://physionet.org/content/vitaldb/1.0.0/vital_files/#files-panel) and download the required vital file.

2. **Configure the File Path**:
   - Place the address of your downloaded vital file in the `config.yaml` file.

3. **Run the Dashboard**:
   - Open the terminal.
   - Navigate to the directory containing `dash.py`.
   - Execute the following command:
     ```
     streamlit run dash.py
     ```
4. **Module.py**:
   - All the modules are in this file.

## Research Question

**How does the intraoperative electrocardiogram (ECG) data correlate with arterial blood pressure readings in surgical patients, and can variations in this relationship predict perioperative cardiovascular complications?**

## Analysis of Correlation Between ECG and ART Data

1. **Time Series Analysis**:
   - The time series plots for ECG_II and ART data display numerous spikes and fluctuations, indicating dynamic variations during surgery.
   - Observing these fluctuations and how they coincide can provide insights into their synchronous behavior, which is crucial for understanding their relationship.

2. **Smoothed Signal Plots**:
   - The smoothed signal plots for both ECG_II and ART using the Savitzky-Golay filter show the overall patterns and trends while reducing noise.
   - These plots allow for easier identification of underlying trends and patterns in both signals, facilitating a clearer analysis of how changes in one signal may correspond to changes in the other.

3. **Cross-Correlation Analysis**:
   - The cross-correlation plot between ECG_II and ART signals at different lags indicates the degree of correlation at various time shifts.
   - A peak at lag 0 suggests a strong synchronous relationship, meaning changes in ECG can directly correspond to changes in ART without any delay.
   - Fluctuations at other lags provide insight into whether the relationship persists over time or if there's a delay in the physiological response between the heart's electrical activity and arterial pressure.

4. **Autocorrelation Analysis**:
   - The autocorrelation plots for both ECG_II and ART help in understanding their temporal dependencies.
   - ECG_II shows a quick decay in autocorrelation, indicating short-term memory, whereas ART maintains a more prolonged autocorrelation, indicating longer-term dependencies.
   - This suggests that while ECG_II changes rapidly, ART changes are more persistent over time.

## Predicting Perioperative Cardiovascular Complications

1. **Rate of Change Analysis**:
   - The rate of change analysis for both signals highlights rapid changes, which can indicate anomalies or critical events.
   - Identifying significant spikes in rate of change can help detect abrupt and potentially dangerous fluctuations in cardiac activity and blood pressure.

2. **Exceedances Analysis**:
   - By analyzing exceedances, we can detect how often the signals surpass critical thresholds, which could indicate periods of instability or risk.
   - Frequent exceedances in ART or ECG_II could correlate with perioperative complications.

3. **Statistical Summaries and Confidence Intervals**:
   - The statistical summaries, including mean, median, standard deviation, skewness, and kurtosis, provide a comprehensive overview of the data distribution.
   - Confidence intervals give an estimate of the precision of these metrics, helping to assess the reliability of the observed patterns.
   - Consistent deviations in these statistics from normal ranges could serve as early warning signs for potential complications.

## Conclusion

By integrating these analyses, we can establish that:
- **Strong synchronous relationships** between ECG_II and ART, indicated by cross-correlation peaks at lag 0, suggest that changes in cardiac electrical activity are directly reflected in arterial pressure changes.
- **Persistent and significant fluctuations** in these signals, identified through time series, smoothed plots, rate of change, and exceedances analyses, can be critical indicators of perioperative cardiovascular complications.
- **Statistical summaries and confidence intervals** provide a solid foundation for understanding the overall stability and variability of these signals during surgery.

These insights can be pivotal in predicting perioperative cardiovascular complications, as variations and anomalies in the relationship between ECG and ART readings can signal underlying issues requiring medical intervention.

## Resources

- [PhysioNet VitalDB](https://physionet.org/content/vitaldb/1.0.0/vital_files/#files-panel)
- stackoverflow.co
- chatgpt.com



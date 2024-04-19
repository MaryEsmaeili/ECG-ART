# ECG-ART
final assignment 3
Research Project: ECG Variability and Arterial Blood Pressure Analysis
Research Question How do fluctuations in ECG signal variability correlate with changes in arterial blood pressure over time within a clinical environment? 
Methods 
Our study leveraged the VitalDB dataset, with a particular focus on 'SNUADC/ECG_II' for ECG signals and 'SNUADC/ART' for arterial blood pressure readings. The analytical process encompassed:
1.	Data Cleaning: 
• Invalid negative readings in the blood pressure data were considered errors and substituted with NaNs to align with clinical expectations. 
• Linear interpolation addressed gaps in data, laying the groundwork for subsequent analyses.
2.	Outlier Management: 
• The IQR method, enhanced with a stringent multiplier, was employed to pinpoint and manage outliers. 
• Boxplots graphically presented these outliers, providing an intuitive visualization.
3.	Signal Processing: 
• We utilized Savitzky-Golay filters to smooth the data, which improved the visibility of trends and dampened noise interference. 
• Peak detection in the frequency domain yielded insights into the periodicity and regularity of the signals.
4.	Time Series Decomposition: 
• We decomposed the ECG signal to distinguish between the trend, seasonal, and residual elements. 
• This was instrumental in dissecting the signal’s intrinsic characteristics over the observation period. 
Outputs 
The research produced pivotal outputs, such as: 
• Boxplots showcasing the spread of both ECG and arterial blood pressure data, with outliers clearly marked. 
• Superimposed plots delineating the initial and smoothed signals, which illustrated the Savitzky-Golay filter’s impact. 
• Spectral analyses of both signals, revealing their dominant frequencies. 
• Decomposition charts for the ECG data, delineating the observed data, underlying trend, seasonal fluctuations, and residuals. 
Conclusion 
The exploration unveiled a multifaceted interaction between ECG variability and arterial blood pressure, signaling directions for deeper analysis that could enrich patient monitoring and the development of prognostic models in healthcare. Our methodology integrated statistical analysis with advanced signal processing, fitting for complex clinical data examination. Reference The data for this study was obtained from the VitalDB dataset, which is available at PhysioNet: VitalDB Dataset.


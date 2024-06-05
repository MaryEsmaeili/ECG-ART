import streamlit as st
import yaml
import pandas as pd
import numpy as np
import io
from module import PrepareVitalSignals, VitalSignalsVisualizer, VitalSignsAnalyzer
from scipy.signal import welch, get_window, find_peaks
from scipy import stats
import matplotlib.pyplot as plt

# Function to display dataframe information
def display_dataframe_info(df, title, st):
    st.subheader(title)
    st.write("### Data Types")
    st.write(df.dtypes)
    st.write("### Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("### Descriptive Statistics")
    st.write(df.describe())
    st.write("### Percentage of Missing Values")
    for column in df.columns:
        st.write(f"{column}: {df[column].isnull().mean() * 100:.2f}%")
    st.write("### Percentage of Negative Values")
    for column in df.select_dtypes(include=[np.number]).columns:  # Ensure only numeric columns are checked
        st.write(f"{column}: {(df[column] < 0).mean() * 100:.2f}%")
    st.write("### Percentage of NaN Values")
    for column in df.columns:
        st.write(f"{column}: {df[column].isna().mean() * 100:.2f}%")


def get_config():
    """
    opens and returns config file
    """
    with open("config.yaml", 'r', encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config

def display_plot_explanations(title, explanation, st):
    st.subheader(title)
    st.write(explanation)

# Set file path and parameters
config = get_config()
file_path = config['data_file_path']
tracks = ["SNUADC/ART", "SNUADC/ECG_II"]  
intervals = 5

# Initialize the PrepareVitalSignals class
prepare_vitals = PrepareVitalSignals(file_path, tracks, intervals)

# Load data
prepare_vitals.load_data()
raw_df = prepare_vitals.df

# Clean the data
cutoff_time_seconds = None  # Replace with your cutoff time if needed
prepare_vitals.clean_data(cutoff_time_seconds)
cleaned_df = prepare_vitals.get_cleaned_data()

# Initialize the VitalSignalsVisualizer class
visualizer = VitalSignalsVisualizer(cleaned_df)

# Initialize the VitalSignsAnalyzer class
analyzer = VitalSignsAnalyzer(cleaned_df, time_window='1T')

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Introduction",
    "Raw Data Information",
    "Cleaned Data Information",
    "Q-Q Plots",
    "ECG and ART Time Series Plots",
    "Summarized Time Frame Data",
    "Smoothed Signal Plots",
    "Moving Average Plots",
    "Seasonal Decomposition Plots",
    "Autocorrelation Analysis",
    "Cross-Correlation Analysis",
    "Nyquist Frequency",
    "Regenerated Signal with Apodization",
    "Peaks and Troughs Detection",
    "Rate of Change Analysis",
    "Exceedances Analysis",
    "95% Confidence Interval",
    "Dangerous Area Detection"
])

# Content based on selection
if section == "Introduction":
    st.title("Vital Signals Dashboard")
    st.write("""
    This dashboard provides an analysis of vital signals including ECG and ART data.
    Use the navigation menu to explore different sections of the analysis.
    """)

elif section == "Raw Data Information":
    display_dataframe_info(raw_df, "Raw Data Information", st)

elif section == "Cleaned Data Information":
    display_dataframe_info(cleaned_df, "Cleaned Data Information", st)

elif section == "Q-Q Plots":
    visualizer.plot_qq(st)
    display_plot_explanations("Plot Explanations", """
        1. **Q-Q Plot for ECG_II**: This Q-Q plot shows the deviation of ECG_II sample quantiles from a theoretical normal distribution, indicating a right-skewed distribution with several outliers.
        2. **Q-Q Plot for ART**: The Q-Q plot for ART demonstrates the sample quantiles compared to a theoretical normal distribution, revealing a right-skewed pattern and significant outliers.
        """, st)

elif section == "ECG and ART Time Series Plots":
    visualizer.plot_ecg_art(st)
    display_plot_explanations("Plot Explanations", """
        3. **ECG_II and ART Time Series Plots**: These time series plots display the variation of ECG_II and ART signals over time, highlighting the presence of numerous spikes and fluctuations in the data.
        """, st)

elif section == "Summarized Time Frame Data":
    st.subheader("Summarized Time Frame Data")
    summary_df = analyzer.summarize_time_frame()
    st.write(summary_df)
    st.subheader("Explanation of Summarized Time Frame Data")
    st.write("""
    The summarized time frame data provides statistical features over the specified time window, including mean, median, standard deviation, minimum, and maximum values. It also includes skewness and kurtosis to describe the distribution shape of each resampled group.
    """)

elif section == "Smoothed Signal Plots":
    st.subheader("Smoothed Signal Plots")
    # Apply Savitzky-Golay filter and plot signals
    for col in ['ECG_II', 'ART']:
        analyzer.apply_savgol_filter(col)
        fig = analyzer.plot_signals(col)
        st.pyplot(fig)
    st.subheader("Explanation of Smoothed Signal Plots")
    st.write("""
    1. **Smoothed Signal Plot for ECG_II**: This plot shows the original ECG_II signal (in blue) overlaid with the smoothed ECG_II signal (in red) using the Savitzky-Golay filter. The original signal exhibits sharp spikes and noise, while the smoothed signal retains the overall pattern but with reduced noise, making it easier to identify trends and patterns. The Savitzky-Golay filter helps in reducing high-frequency noise while preserving the shape and features of the signal, which is crucial for accurate analysis.
    
    2. **Smoothed Signal Plot for ART**: This plot displays the original ART signal (in blue) overlaid with the smoothed ART signal (in red) using the Savitzky-Golay filter. The original ART signal contains significant spikes and fluctuations, which are smoothed out in the red line, making the underlying trends more apparent. This demonstrates the effectiveness of the Savitzky-Golay filter in smoothing arterial pressure signals, which helps in reducing noise and highlighting the essential characteristics of the signal for further analysis.
    """)


elif section == "Moving Average Plots":
    st.subheader("Moving Average Plots")
    window_size = 10  # Define your window size
    for col in ['ECG_II', 'ART']:
        analyzer.moving_average(col, window_size)
        st.write(f"### Moving Average for {col}")
        st.line_chart(analyzer.df[[col, f'smooth_{col}']])
    st.subheader("Explanation of Moving Average Plots")
    st.write("""
    1. **Moving Average Plots for ART**: This plot shows the original ART signal (in dark blue) overlaid with the smoothed ART signal (in light blue) using the moving average technique. The original signal contains significant spikes and fluctuations, while the smoothed signal provides a clearer view of the overall trend by reducing short-term noise. This smoothing helps in identifying underlying patterns and trends in arterial pressure, which are critical for analyzing cardiovascular health.
    
    2. **Moving Average Plots for ECG_II**: This plot displays the original ECG_II signal (in dark blue) overlaid with the smoothed ECG_II signal (in light blue) using the moving average technique. The original signal exhibits sharp spikes and noise, while the smoothed signal retains the overall pattern but with reduced noise. This makes it easier to identify trends and patterns in the heart's electrical activity, aiding in the detection of anomalies and providing insights into cardiac function.
    """)

elif section == "Seasonal Decomposition Plots":
    st.subheader("Seasonal Decomposition Plots")
    window_size = 50  # Define your window size for seasonal decomposition
    for col in ['ECG_II', 'ART']:
        st.write(f"### Seasonal Decomposition for {col}")
        fig = analyzer.get_seasonal_decompose(col, window_size)
        st.pyplot(fig)
    st.subheader("Explanation of Seasonal Decomposition Plots")
    st.write("""
    1. **Seasonal Decomposition for ECG_II**: This plot breaks down the ECG_II signal into three components: trend, seasonal, and residual.
        - **Trend**: The trend component shows the long-term movement in the ECG_II data, indicating an overall pattern of fluctuations.
        - **Seasonal**: The seasonal component captures the repeating cycles or patterns at regular intervals, highlighting periodic behavior in the ECG_II signal.
        - **Residual**: The residual component represents the noise or irregular variations that are not explained by the trend or seasonal components. Understanding these components helps in better analysis and forecasting.
    
    2. **Seasonal Decomposition for ART**: This plot decomposes the ART signal into trend, seasonal, and residual components.
        - **Trend**: The trend component shows the long-term progression in the ART data, indicating overall increases or decreases in arterial pressure over time.
        - **Seasonal**: The seasonal component identifies repeating cycles or patterns in the ART data, capturing regular fluctuations due to physiological processes.
        - **Residual**: The residual component contains the noise or random variations not explained by the trend or seasonal components. Understanding these components helps in separating and analyzing different aspects of the ART signal, such as identifying cyclic patterns and removing noise for improved signal analysis.
    """)


elif section == "Autocorrelation Analysis":
    st.subheader("Autocorrelation Analysis")
    max_lag = 50  # Define your max lag
    for col in ['ECG_II', 'ART']:
        st.write(f"### Autocorrelation for {col}")
        autocorrs = analyzer.autocorrelation_analysis(col, max_lag)
        st.line_chart(autocorrs)
    st.subheader("Explanation of Autocorrelation Analysis")
    st.write("""
    1. **Autocorrelation Plot for ECG_II**: This plot displays the autocorrelation coefficients of the ECG_II signal at different lags. The initial high autocorrelation at lag 0 (1.0) quickly diminishes as the lag increases, indicating that the signal values are highly correlated with themselves at short time lags but this correlation weakens at longer lags. This suggests that the ECG_II signal has a short memory, meaning that its current values are more influenced by recent past values rather than older ones.
    
    2. **Autocorrelation Plot for ART**: This plot shows the autocorrelation coefficients of the ART signal at different lags. Similar to the ECG_II plot, the autocorrelation starts high at lag 0 and decreases as the lag increases, though it remains above zero for a longer range of lags compared to ECG_II. This indicates that the ART signal has a more persistent correlation over time, suggesting that arterial pressure values have a longer memory and are influenced by a longer history of past values. Understanding this characteristic helps in selecting appropriate models for analyzing and forecasting the ART signal.
    """)

elif section == "Cross-Correlation Analysis":
    st.subheader("Cross-Correlation Analysis")
    max_lag = 50  # Define your max lag
    for col1, col2 in [('ECG_II', 'ART')]:
        st.write(f"### Cross-Correlation between {col1} and {col2}")
        cross_corrs = analyzer.cross_correlation_analysis(col1, col2, max_lag)
        lags = range(-max_lag, max_lag + 1)
        cross_corr_df = pd.DataFrame({'lag': lags, 'cross_correlation': cross_corrs})
        st.line_chart(cross_corr_df.set_index('lag'))
    st.subheader("Explanation of Cross-Correlation Analysis")
    st.write("""
    1. **Cross-Correlation Plot between ECG_II and ART**: This plot shows the cross-correlation coefficients between the ECG_II and ART signals at different lags. The peak at lag 0 indicates the highest correlation between the two signals when there is no lag, suggesting a synchronous relationship. The fluctuations at other lags show how the correlation changes when one signal is shifted in time relative to the other. This helps in understanding the temporal relationship between the ECG_II and ART signals, indicating whether changes in one signal can be used to predict changes in the other.
    """)

elif section == "Nyquist Frequency":
    st.subheader("Nyquist Frequency")
    for col in ['ECG_II', 'ART']:
        nyquist_freq = analyzer.find_Nyquist_frequency(col)
        st.write(f"The Nyquist frequency of the {col} signal is {nyquist_freq:.2f} Hz.")
    st.subheader("Explanation of Nyquist Frequency")
    st.write("""
    The Nyquist frequency is the highest frequency that can be accurately sampled. It is half of the sampling rate of the signal.
    """)

elif section == "Regenerated Signal with Apodization":
    st.subheader("Regenerated Signal with Apodization")
    window_type = 'hann'  # Example window type, you can change this or make it user-selectable
    for col in ['ECG_II', 'ART']:
        st.write(f"### Regenerated Signal for {col} using {window_type} window")
        signal_length = len(cleaned_df[col].dropna())
        zeros_num = st.number_input(f"Number of zeros to pad for {col} (must be larger than {signal_length})", min_value=signal_length + 1, value=signal_length + 100)
        if zeros_num > signal_length:
            regenerated_signal = analyzer.regenerate_signal_with_apodization(col, window_type, zeros_num)
            regenerated_df = pd.DataFrame({'Time': range(len(regenerated_signal)), 'Regenerated Signal': regenerated_signal})

            # Plot the regenerated signal
            fig, ax = plt.subplots()
            ax.plot(regenerated_df['Time'], regenerated_df['Regenerated Signal'])
            ax.set_title(f'Regenerated Signal for {col}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            st.pyplot(fig)
    st.subheader("Explanation of Regenerated Signal with Apodization")
    st.write("""
    1. **Regenerated Signal for ECG_II**: This plot shows the regenerated ECG_II signal after applying the Hann window and padding with zeros. The signal has been transformed using the Fourier transform and then inverted back, demonstrating a clear reduction in artifacts and edge effects. The symmetric shape around the center indicates the successful application of apodization, which mitigates the effects of Gibb's phenomenon. This results in a cleaner frequency representation of the ECG_II signal, enabling more accurate analysis of its frequency components.
    
    2. **Regenerated Signal for ART**: This plot displays the regenerated ART signal after applying the Hann window and padding with zeros. The transformation through the Fourier domain and back helps to reduce discontinuities and artifacts, resulting in a smoother signal with a prominent central peak. This indicates that the arterial pressure signal's frequency components are well-preserved while minimizing edge effects. Such techniques ensure that the signal analysis is accurate and free from artifacts that could misrepresent the underlying physiological processes.
    """)

elif section == "Peaks and Troughs Detection":
    st.subheader("Peaks and Troughs Detection")
    for col in ['ECG_II', 'ART']:
        st.write(f"### Peaks and Troughs for {col}")
        peaks, troughs = analyzer.find_peaks_and_troughs(col)
        st.write(f"Peaks: {peaks}")
        st.write(f"Troughs: {troughs}")
        st.subheader("Peaks and Troughs Detection")
    
        st.write(f"### Peaks and Troughs for {col}")
        peaks, troughs = analyzer.find_peaks_and_troughs(col)
        st.write(f"Peaks: {peaks}")
        st.write(f"Troughs: {troughs}")
        median_peaks, median_troughs = analyzer.find_median_peaks_and_troughs(col)
        st.write(f"Median of Peaks: {median_peaks}")
        st.write(f"Median of Troughs: {median_troughs}")
    st.subheader("Explanation of Peaks and Troughs Detection")
    st.write("""
    Detecting peaks and troughs in the signal helps to identify significant points in the signal's waveform. Peaks represent local maxima, and troughs represent local minima, which are useful for understanding the signal's behavior over time.
    """)

elif section == "Rate of Change Analysis":
    st.subheader("Rate of Change Analysis")
    for col in ['ECG_II', 'ART']:
        st.write(f"### Rate of Change for {col}")
        rate_of_change = analyzer.calculate_rate_of_change(col, periods=1)
        rate_of_change_df = pd.DataFrame({f'Rate of Change of {col}': rate_of_change})
        st.line_chart(rate_of_change_df)
    st.subheader("Explanation of Rate of Change Analysis")
    st.write("""
    1. **Rate of Change for ECG_II**: This plot shows the rate of change for the ECG_II signal. The rate of change is calculated as the difference in signal values over successive time points, highlighting rapid changes in the ECG_II readings. Spikes in the plot represent sudden changes in the heart's electrical activity, which could be indicative of anomalies or significant events. Analyzing the rate of change helps in detecting abrupt changes, trends, and potential irregularities in the ECG_II signal.
    
    2. **Rate of Change for ART**: This plot illustrates the rate of change for the ART (arterial pressure) signal. Similar to the ECG_II plot, the rate of change here shows the differences in ART values over time. Significant spikes indicate sudden changes in arterial pressure, which could be crucial for identifying periods of high variability or instability in the patient's blood pressure. This analysis aids in monitoring and detecting critical fluctuations in arterial pressure that may require medical attention.
    """)

elif section == "Exceedances Analysis":
    st.subheader("Exceedances Analysis")
    threshold = 100  # Example threshold, you can change this or make it user-selectable
    for col in ['ECG_II', 'ART']:
        count = analyzer.count_exceedances(col, threshold)
        st.write(f"The number of times the {col} signal exceeds the threshold of {threshold} is {count}.")
    st.subheader("Explanation of Exceedances Analysis")
    st.write("""
    The exceedances analysis counts the number of times the signal exceeds a specified threshold. This can be useful for detecting abnormal events or conditions that need attention.
    """)

elif section == "95% Confidence Interval":
    st.subheader("95% Confidence Interval")
    for col in ['ECG_II', 'ART']:
        mean, confidence_interval = analyzer.calculate_95_percent_confidence_interval(col)
        st.write(f"The 95% confidence interval for the mean of the {col} signal is {confidence_interval} with a mean of {mean:.2f}.")
    st.subheader("Explanation of 95% Confidence Interval")
    st.write("""
    1. **95% Confidence Interval for ECG_II**: The 95% confidence interval for the mean of the ECG_II signal is (0.1257, 0.1321) with a mean of 0.13. This interval suggests that we can be 95% confident that the true mean of the ECG_II signal lies within this range. The mean value of 0.13 indicates the average level of the ECG_II signal, while the confidence interval provides an estimate of the precision of this mean. Understanding the confidence interval helps in assessing the reliability of the mean estimate and the variability in the ECG_II signal.
    
    2. **95% Confidence Interval for ART**: The 95% confidence interval for the mean of the ART signal is (69.85, 71.21) with a mean of 70.53. This interval indicates that we can be 95% confident that the true mean of the ART signal falls within this range. The mean value of 70.53 represents the average arterial pressure, and the confidence interval gives an estimate of the precision of this mean. This information is crucial for evaluating the reliability of the mean arterial pressure and understanding the extent of variability in the ART signal.
    """)

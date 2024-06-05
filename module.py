import vitaldb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.signal.windows import get_window
import scipy.stats as st
import statsmodels.api as sm
import io 

class PrepareVitalSignals:
    """
    This class loads, cleans, and addresses anomalies in vital signs data.
    """
    def __init__(self, file_path, tracks, intervals):
        self.file_path = file_path
        self.tracks = tracks
        self.intervals = intervals
        self.df = None

    def load_data(self):
        """Load the data using the VitalDB library."""
        vf = vitaldb.VitalFile(self.file_path)
        self.df = vf.to_pandas(self.tracks, interval=self.intervals, return_datetime=True)

    def save_to_csv(self, csv_file_path):
        """Save the DataFrame to a CSV file."""
        if self.df is not None:
            self.df.to_csv(csv_file_path, index=False)
            print(f"Data saved to {csv_file_path}")
        else:
            print("DataFrame is empty. Please load the data first.")

    def get_data_types(self):
        """Return the data types of all columns in the DataFrame."""
        return self.df.dtypes

    def print_dataset_info(self):
        """Print the structure and summary statistics of the DataFrame."""
        if self.df is not None:
            print(self.df.info())
            print(self.df.describe())
        else:
            print("DataFrame is empty. Please load the data first.")

    def percentage_missing_values(self, column):
        """Return the percentage of missing values in the specified column."""
        null_value = self.df[column].isnull().sum()
        return (null_value / len(self.df[column])) * 100
         
    def percentage_negative_values(self, column):
        """Return the percentage of negative values in the specified column."""
        neg_value = (self.df[column] < 0).sum()
        return (neg_value / len(self.df[column])) * 100

    def percentage_nan_values(self, column):
        """"Return the percentage of NaN values in the specified column."""
        nan_value = self.df[column].isna().sum()
        return (nan_value / len(self.df[column])) * 100

    def clean_data(self, cutoff_time_seconds=None):
        """
        Clean the data by converting 'Time' to a timedelta format, setting it as the index,
        converting object columns to numeric, replacing negative values with NaN, 
        renaming columns, and interpolating missing values.

        Parameters:
        - cutoff_time_seconds: Optional; if provided, data before this time will be excluded.
        """
        # Convert the 'Time' column to datetime
        self.df['Time'] = pd.to_datetime(self.df['Time'])

        # Convert 'Time' from datetime to total seconds since the start
        self.df['Time'] = (self.df['Time'] - self.df['Time'].min()).dt.total_seconds()

        # Set 'Time' column as the index
        self.df.set_index(self.df['Time'], inplace=True)

        # Convert object columns to numeric and coerce errors to NaN
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Replace negative values with NaN so we can fill them with interpolation
            self.df.loc[self.df[col] < 0, col] = np.nan

        # Rename columns to simpler names for easier reference
        new_column_names = {
            'SNUADC/ART': 'ART',
            'SNUADC/ECG_II': 'ECG_II',
            # Add more column renaming as needed
        }
        self.df.rename(columns=new_column_names, inplace=True)

        # Interpolate missing values linearly
        self.df.interpolate(method='linear', inplace=True)

        # Optionally, exclude data before a certain time
        if cutoff_time_seconds is not None:
            self.df = self.df[self.df.index > cutoff_time_seconds]

    def get_outliers_with_describe(self, column1, column2):
        """
        Returns descriptive statistical data of our dataframe for specified columns.
        
        Parameters:
        - col1: The name of the first column.
        - col2: The name of the second column.
        
        Returns:
        - A DataFrame containing descriptive statistics for the specified columns.
        """
        return self.df.describe()[[column1, column2]]

    def get_outliers_with_threshold(self, column, *, threshold=3):
        """
        Identifies outliers using the specified threshold method (common value for threshold is 3).
        
        Parameters:
        - col: The column name to check for outliers.
        - threshold: The number of standard deviations to use as the threshold.
        
        Returns:
        - A DataFrame containing the outliers.
        """
        mean_col = self.df[column].mean()
        std_col = self.df[column].std()

        # Find outliers
        outliers = self.df[(self.df[column] > mean_col + threshold * std_col) | 
                            (self.df[column] < mean_col - threshold * std_col)]
        return outliers


    def find_outliers_iqr(self, column):
        """
        Identifies outliers in a DataFrame column using the Interquartile Range (IQR) method.

        Parameters:
        - column: The exact column name in which to find outliers.

        Returns:
        - A DataFrame containing the outliers.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define outliers as those values that fall outside of Q1 - 1.5*IQR and Q3 + 1.5*IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers


    def remove_outliers(self):
        """
        Remove outliers from the DataFrame based on IQR for each column.
        """
        for column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]


    def get_cleaned_data(self):
        """Returns the cleaned DataFrame"""
        return self.df



class VitalSignsAnalyzer:
    """
    This class performs feature extraction and analysis on vital signals data.
    """
    def __init__(self, df, time_window):
        """
        Initialize the VitalSignsAnalyzer with a DataFrame and time window.

        Parameters:
        - df : pandas.DataFrame
          DataFrame containing the signal data with a datetime index.
        - time_window : str
          Pandas offset alias.
        """
        self.df = df
        self.time_window = time_window
    
    def summarize_time_frame(self):
        """
        Summarizes the signal data over the specified time frame.

        Returns:
        - summary_df : pandas.DataFrame
          DataFrame containing summarized features for each time frame.
        """
        # Ensure the index is a DatetimeIndex for resampling
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index, unit='s', origin='unix')

        # Initialize a dictionary to hold our summarized data
        summary_dict = {}

        # Resample data over the specified time window and calculate summary statistics
        resampled_df = self.df.resample(self.time_window)
        summary_dict['mean'] = resampled_df.mean()
        summary_dict['median'] = resampled_df.median()
        summary_dict['std_dev'] = resampled_df.std()
        summary_dict['min'] = resampled_df.min()
        summary_dict['max'] = resampled_df.max()

        # Calculate skewness and kurtosis for each resampled group
        for column in self.df.columns:
            grouped = resampled_df[column]
            summary_dict[f'{column}_skewness'] = grouped.agg(lambda x: stats.skew(x.dropna()))
            summary_dict[f'{column}_kurtosis'] = grouped.agg(lambda x: stats.kurtosis(x.dropna()))
        
        # Convert dictionary to DataFrame
        summary_df = pd.concat(summary_dict, axis=1)
        return summary_df

    def apply_savgol_filter(self, col, window_length=5, polyorder=2):
        """
        The Savitzky-Golay filter reduces noise in vital signals (like ECG and ART) while preserving essential features,
        and its adjustable parameters allow for tailored smoothing.
        Parameters:
        - col : str
          The column to smooth.
        - window_length : int
          The length of the filter window (default is 5).
        - polyorder : int
          The order of the polynomial used to fit the samples (default is 2).
        """
        self.df[f'SG_Smoothed_{col}'] = savgol_filter(self.df[col], window_length, polyorder)

    def plot_signals(self, col, start_index=0):
        """
        Parameters:
        - col : str
          The column to plot.
        - start_index : int
          The start index of the range to plot (default is 0).
        - end_index : int
          The end index of the range to plot (default is 500).
        """
        plt.figure(figsize=(14, 10))
        plt.plot(self.df.index[start_index:], self.df[col][start_index:], label=f'Original {col}', color='blue')
        plt.plot(self.df.index[start_index:], self.df[f'SG_Smoothed_{col}'][start_index:], label=f'Smoothed {col}', color='red', linestyle='--')
        plt.title(f'{col} Signal: Original and Smoothed')
        plt.xlabel('Time Points')
        plt.ylabel(f'{col} Reading')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def moving_average(self, col , window_size):
        self.df[f'smooth_{col}']=self.df[col].rolling(window=window_size).mean()

    def get_seasonal_decompose(self, col, window_size):
        """Applies decomposition on given signal"""
        decompose = seasonal_decompose(self.df[col].dropna(), model='additive', period=window_size)
        decompose.plot()
        plt.show()

    def autocorrelation_analysis(self, col, max_lag):
        """
        Calculates the autocorrelation for different lags for every column.

        Parameters:
        - col : str
        The column to calculate autocorrelation for.
        - max_lag : int
        The maximum lag to calculate autocorrelation for.

        Returns:
        - autocorrs : pandas.Series
        Autocorrelation coefficients for each lag up to max_lag.
        """
        autocorrs = pd.Series([self.df[col].autocorr(lag) for lag in range(max_lag + 1)])
        return autocorrs

    def cross_correlation_analysis(self, col1, col2, max_lag):
        """
        Calculates the cross-correlation between columns for different lags.

        Parameters:
        - col1, col2 : str
        The columns to calculate cross-correlation between.
        - max_lag : int
        The maximum lag to calculate cross-correlation for.

        Returns:
        - cross_corrs : numpy.array
        Cross-correlation coefficients for each lag up to max_lag.
        """
        cross_corrs = [self.df[col1].corr(self.df[col2].shift(lag)) for lag in range(-max_lag, max_lag + 1)]
        return np.array(cross_corrs)

    def frequency_analysis(self, col):
        """
        Using Welch's method.
        Welch's method estimates the power spectral density of the signal, handling noise effectively and 
        providing accurate frequency component analysis essential for 
        diagnosing conditions from vital signals like ECG and ART.

        Parameters:
        - col : str
          The column to analyze.

        Returns:
        - freqs : numpy.array
          Array of sample frequencies.
        - psd : numpy.array
          Power spectral density of the signal.
        """
        # Get the signal values
        signal_values = self.df[col].dropna().values

        # Compute the Power Spectral Density (PSD)
        freqs, psd = welch(signal_values, fs=1/(self.df.index[1] - self.df.index[0]).total_seconds())

        return freqs, psd

    def find_period_of_signal(self, col):
        """
        Parameters:
        - col : str
          The column to analyze.
        """
        ds = (np.max(self.df[col]) - np.min(self.df[col])) / len(self.df[col])
        spectrum = np.fft.fft(self.df[col], norm='forward')
        power_spectrum = np.abs(spectrum)**2
        frequencies = np.fft.fftfreq(len(self.df[col]), d=ds)
        positive_frequencies = frequencies[1:int(len(frequencies)/2)]
        positive_power = power_spectrum[1:int(len(power_spectrum)/2)]
        dominant_frequency = positive_frequencies[np.argmax(positive_power)]
        return 1/dominant_frequency

    def find_Nyquist_frequency(self, col):
        """
        Parameters:
        - col : str
          The column to analyze.
        """
        ds = (np.max(self.df[col]) - np.min(self.df[col])) / len(self.df[col])
        return 0.5 * 1/ds
    
    def regenerate_signal_with_apodization(self, col, window, zeros_num):
        """
        The regenerated signal, after applying apodization and padding, gives a better representation of 
        the underlying physiological phenomena without being distorted by the edge effects.
        For tasks such as detecting periodicities, dominant frequencies, or analyzing 
        the frequency content of the signals, this method provides a more reliable result.
        
        Parameters:
        - col : str
        The column to analyze.
        - window : str
        the name of window from this list('boxcar','triang','blackman','hamming','hann','bartlett','flattop','parzen','bohman','blackmanharris','nuttall','barthann','cosine','tukey','taylor','lanczos')
        Returns:
        - regenerated_signal : numpy.array
        Array of regenerated signal.
        """
        if zeros_num < len(self.df[col]):
            raise ValueError("zeros_num must be larger than the length of the signal")

        signal = self.df[col].dropna().values
        windowed_signal = signal * get_window(window, len(signal))
        num_zeros = zeros_num - len(signal)
        y = np.concatenate((windowed_signal, np.zeros(num_zeros)))
        c = np.fft.fftshift(np.fft.fft(y, norm='forward'))
        regenerated_signal = np.fft.ifft(np.fft.ifftshift(c), norm='forward').real

        return regenerated_signal

    
    def find_peaks_and_troughs(self, column='HR', height=None, distance=None, threshold=None):
        peaks, _ = find_peaks(self.df[f'smooth_{column}'], height=height, distance=distance, threshold=threshold)
        troughs, _ = find_peaks(-self.df[f'smooth_{column}'], height=height, distance=distance, threshold=threshold)
        return peaks, troughs
    
    def find_median_peaks_and_troughs(self, column='HR', height=None, distance=None, threshold=None):
        peaks, troughs = self.find_peaks_and_troughs(column, height, distance, threshold)
        
        if len(peaks) > 0:
            median_peaks = np.median(peaks)
        else:
            median_peaks = None

        if len(troughs) > 0:
            median_troughs = np.median(troughs)
        else:
            median_troughs = None

        return median_peaks, median_troughs

    def calculate_rate_of_change(self, column, periods=1, plot=False):
        rate_of_change = self.df[column].diff(periods=periods) / periods
        if plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(self.df[column], label=f'{column} Signal')
            plt.title(f'{column} Signal')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(rate_of_change, label=f'Rate of Change of {column}', color='red')
            plt.title(f'Rate of Change of {column}')
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return rate_of_change

    def count_exceedances(self, column, threshold):
        exceedances = self.df[column] > threshold
        count = exceedances.sum()
        return count

    def calculate_95_percent_confidence_interval(self, col):
        mean = self.df[col].mean()
        std = self.df[col].std()
        sem = std / np.sqrt(len(self.df[col]))
        confidence_interval = stats.t.interval(0.95, len(self.df[col])-1, loc=mean, scale=sem)
        return mean, confidence_interval


class VitalSignalsVisualizer:
    """
    This class handles visualization of vital signals data.
    """
    def __init__(self, df):
        self.df = df

    def plot_qq(self, st):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column.lower() != 'time':  # Exclude 'Time' column
                st.write(f"### Q-Q Plot for {column}")
                fig = sm.qqplot(self.df[column].dropna(), line='s')
                plt.title(f"Q-Q Plot for {column}")
                st.pyplot(fig)

    def plot_ecg_art(self, st):
        st.subheader("ECG and ART Plots")
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot ECG
        if 'ECG_II' in self.df.columns:
            axs[0].plot(self.df.index, self.df['ECG_II'])
            axs[0].set_title('ECG_II')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('ECG_II')

        # Plot ART
        if 'ART' in self.df.columns:
            axs[1].plot(self.df.index, self.df['ART'])
            axs[1].set_title('ART')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('ART')

        st.pyplot(fig)

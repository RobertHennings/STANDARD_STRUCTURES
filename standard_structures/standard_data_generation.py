from typing import List, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class StandardDataGeneration(object):
    def __init__(
            self
            ):
        pass

    def generate_ar_1_process(
            self,
            phi: float,
            starting_value: float,
            constant_c: float,
            error_term: np.array,
            error_term_variance: float,
            n: int,
            process_len: int,
            return_theoretical_acf: bool=True
            ) -> np.array:
        """Generation of n AR(1) processes with the process length
           process_len with the same parameters.

        Args:
            phi (float): Autocorrelation, normed in between the range [-1, 1]
            starting_value (float): Starting value for the processes
            constant_c (float): constant term
            error_term (np.array): free designable error term
            error_term_variance (float): variance of the error term
            n (int): number of AR(1) processes that shall be generated
            process_len (int): Length of each individual of the n AR(1) processes

        Raises:
            ValueError: _description_

        Returns:
            np.array: n AR(1) processes with same length process_len
        """
        # Check for aligning dimensions
        if error_term.shape[0] != process_len or error_term.shape[1] != n:
            raise ValueError("Error term dimensions do not match the specified process length and number of variables.")
        Y_t = np.zeros((process_len, n))
        Y_t[0, :] = starting_value
        for t in range(1, process_len):
            Y_t[t, :] = constant_c + phi * Y_t[t - 1, :] + constant_c + error_term[t, :]
        # Also compute and return the unconditional first two moments
        unconditional_mean = starting_value + constant_c / (1 - phi)
        unconditional_variance = error_term_variance / (1 - phi ** 2)
        print(f"Unconditional Mean: {unconditional_mean}\nUnconditional Variance: {unconditional_variance}")
        if return_theoretical_acf:
            # Compute the theoretical autocorrelation function (ACF)
            theo_acf_values = [phi**k for k in range(process_len)]
            return Y_t, unconditional_mean, unconditional_variance, theo_acf_values
        else:
            return Y_t, unconditional_mean, unconditional_variance

    def generate_strong_white_noise_process(
            self
            ):
        pass


    def generate_weak_white_noise_process(
            self
            ):
        pass


    def generate_martingale_difference_process(
            self
            ):
        pass

standard_data_generation_instance = StandardDataGeneration()
process_len = 100
n = 10
error_term_variance = 1
phi = 0.5
ar_1_processes, unconditional_mean, unconditional_variance, theo_acf_values = standard_data_generation_instance.generate_ar_1_process(
    phi=phi,
    starting_value=0,
    constant_c=0,
    error_term=np.random.normal(0, error_term_variance, (process_len, n)),
    error_term_variance=error_term_variance,
    n=n,
    process_len=process_len
    )
print(f"Shape of the created AR(1) process array: {ar_1_processes.shape}")
# Output: (100, 10)
columns = [f"AR_1_Process_{i}" for i in range(1, n + 1)]
ar_1_processes_df = pd.DataFrame(
    data=ar_1_processes,
    columns=columns
    )
import os
os.chdir(r"/Users/Robert_Hennings/Projects/STANDARD_STRUCTURES/standard_structures")
from standard_data_plotting import StandardPlotting
PROJECT_PATH = r""
RESULTS_PATH_ENDING = r"results/"
GRAPHS_PATH_ENDING = r"graphs/"
CAU_COLOR_SCALE = ["#9b0a7d", "grey", "black", "grey"]
COLOR_DISCRETE_SEQUENCE_DEFAULT = CAU_COLOR_SCALE
standard_plotting_instance = StandardPlotting(
    project_path=PROJECT_PATH,
    results_path_ending=RESULTS_PATH_ENDING,
    graphs_path_ending=GRAPHS_PATH_ENDING,
    color_discrete_sequence_default=COLOR_DISCRETE_SEQUENCE_DEFAULT
    )
# Plot the AR(1) processes
ar_1_processes_chart = standard_plotting_instance.get_line_chart(
    data=ar_1_processes_df,
    variable_list=ar_1_processes_df.columns,
    title=f"{n} AR(1) Processes of length: {process_len}",
    xaxis_title="Period Index",
    yaxis_title="AR(1) Process Value",
    legend_title="Process Number"
    )
# Add the unconditional mean
ar_1_processes_chart.add_traces(
    go.Scatter(
        x=ar_1_processes_df.index,
        y=[unconditional_mean] * process_len,
        mode="lines",
        name=f"Unconditional Mean: {round(unconditional_mean, 3)}",
        line=dict(color="red", width=2),
        showlegend=True
    )
)
ar_1_processes_chart.show(renderer="browser")
# Plot the theoretical ACF values
theo_acf_values_df = pd.DataFrame(
    data=theo_acf_values,
    columns=[f"ACF for phi: {phi}"]
    )
ar_1_processes_theo_acf_bar_chart = standard_plotting_instance.get_bar_chart(
    data=theo_acf_values_df,
    variable_list=theo_acf_values_df.columns,
    title=f"Theoretical ACF Values of AR(1) processes with phi: {phi}",
    xaxis_title="Period Index",
    yaxis_title="Theoretical ACF Value",
    )
ar_1_processes_theo_acf_bar_chart.show(renderer="browser")


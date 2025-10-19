from typing import List, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_html
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.figure_factory import create_distplot


class StandardPlotting(object):
    def __init__(
            self,
            project_path: str,
            results_path_ending: str,
            graphs_path_ending: str,
            color_discrete_sequence_default: List[str]
            ):
        self.color_discrete_sequence_default = color_discrete_sequence_default
        self.project_path = project_path
        self.results_path_ending = results_path_ending
        self.graphs_path_ending = graphs_path_ending


    def __check_path_existence(
        self,
        path: str
        ):
        """Internal helper method - serves as generous path existence
           checker when saving and reading of an kind of data from files
           suspected at the given location
           
           !!!!If given path does not exist it will be created!!!!

        Args:
            path (str): full path where expected data is saved
        """
        folder_name = path.split("/")[-1]
        path = "/".join(path.split("/")[:-1])
        # FileNotFoundError()
        # os.path.isdir()
        if folder_name not in os.listdir(path):
            print(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            print(f"Folder: {folder_name} created in path: {path}")


    def get_line_chart(
        self,
        data: pd.DataFrame,
        variable_list: List[str],
        color_discrete_sequence: List[str]=None,
        title: str="",
        xaxis: str=None,
        xaxis_title: str="Date",
        yaxis_title: str="Value",
        save_graph: bool=False,
        file_name: str=None,
        file_path: str=None,
        return_html_version: bool=False,
        showlegend: bool=True,
        opacity: float=1.0,
        legend_title: str="",
        height: float=None,
        width: float=None,
        ) -> go.Figure:
        if data.empty:
            raise Exception(f"Provided empty pd.DataFrame: {data}")

        # Set the default color sequence as variable to use
        if color_discrete_sequence is None:
            color_discrete_sequence = self.color_discrete_sequence_default

        # Extract the x-axis data
        if (xaxis is not None and xaxis in data.columns):
            x = data[xaxis]
            # since we loop through the columns of the data, make sure to exclude the
            # respective column
            data = data.drop(xaxis, axis=1)
        elif (xaxis not in data.columns):
            print(f"Chosen xaxis variable: {xaxis} not present in data.columns: {data.columns}")
            x = data.index
        else:
            x = data.index

        fig = go.Figure()
        for i, var in enumerate(variable_list):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=data[var],
                    mode='lines',
                    name=str(var),
                    opacity=opacity,
                    showlegend=showlegend,
                    line=dict(
                        color=color_discrete_sequence[i % len(color_discrete_sequence)],
                        width=2.0                            
                        )
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title=dict(text=legend_title),
            height=height,
            width=width
        )

        html_fig = to_html(fig)
        print("FUNCTION: get_line_chart")
        if save_graph:
            if file_name is not None:
                # either take the given path of the function or use the default specified one
                if file_path is not None:
                    # use the user specified in the function
                    graph_full_save_path = rf'{file_path}/{file_name}'
                else:
                    graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
            # save the figure
            path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
            self.__check_path_existence(path=path_till_folder_name)
            fig.write_html(graph_full_save_path)
        if return_html_version:
            return fig, html_fig
        else:
            return fig


    def get_dual_axis_line_chart(
        self,
        data: pd.DataFrame,
        variable_list: List[str],
        secondary_yaxis_variables: List[str] = None,
        color_discrete_sequence: List[str] = None,
        title: str = "",
        xaxis: str = None,
        xaxis_title: str = "Date",
        yaxis_title: str = "Primary Y-Axis",
        secondary_yaxis_title: str = "Secondary Y-Axis",
        save_graph: bool = False,
        file_name: str = None,
        file_path: str = None,
        return_html_version: bool = False,
        showlegend: bool = True,
        opacity: float = 1.0,
        legend_title: str = "",
        height: float = None,
        width: float = None,
    ) -> go.Figure:
        if data.empty:
            raise Exception(f"Provided empty pd.DataFrame: {data}")

        # Set the default color sequence as variable to use
        if color_discrete_sequence is None:
            color_discrete_sequence = self.color_discrete_sequence_default

        # Extract the x-axis data
        if xaxis is not None and xaxis in data.columns:
            x = data[xaxis]
            data = data.drop(xaxis, axis=1)  # Exclude x-axis column from data
        else:
            x = data.index

        fig = go.Figure()

        # Plot variables on the primary y-axis
        for i, var in enumerate(variable_list):
            if secondary_yaxis_variables and var in secondary_yaxis_variables:
                continue  # Skip variables meant for the secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=data[var],
                    mode='lines',
                    name=str(var),
                    opacity=opacity,
                    showlegend=showlegend,
                    line=dict(
                        color=color_discrete_sequence[i % len(color_discrete_sequence)],
                        width=2.0
                    )
                )
            )

        # Plot variables on the secondary y-axis
        if secondary_yaxis_variables:
            for i, var in enumerate(secondary_yaxis_variables):
                if var not in data.columns:
                    print(f"Variable '{var}' not found in data columns. Skipping.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=data[var],
                        mode='lines',
                        name=f"{var} (r.h.)",
                        opacity=opacity,
                        showlegend=showlegend,
                        line=dict(
                            color=color_discrete_sequence[(len(variable_list) + i) % len(color_discrete_sequence)],
                            width=2.0
                        ),
                        yaxis="y2"  # Assign to secondary y-axis
                    )
                )

        # Update layout with dual y-axes
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis=dict(title=yaxis_title),
            yaxis2=dict(
                title=secondary_yaxis_title,
                overlaying="y",
                side="right"
            ),
            legend_title=dict(text=legend_title),
            height=height,
            width=width
        )

        html_fig = to_html(fig)
        print("FUNCTION: get_dual_axis_line_chart")
        if save_graph:
            if file_name is not None:
                # either take the given path of the function or use the default specified one
                if file_path is not None:
                    graph_full_save_path = rf'{file_path}/{file_name}'
                else:
                    graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
        if return_html_version:
            return fig, html_fig
        else:
            return fig


    def get_heatmap_chart(
            self,
            data: pd.DataFrame,
            variable_list: List[str],
            title: str="",
            xaxis_title: str="",
            yaxis_title: str="",
            colorscale: str="PiYG",
            colorbar_title: str="Linear Correlation",
            zmin: float=None,
            zmax: float=None,
            round_values: int=2,
            text_font_size: int=None,
            show_data_values: bool=True,
            save_graph: bool=False,
            file_name: str=None,
            file_path: str=None,
            return_html_version: bool=False,
            showscale: bool=True,
            height: float=None,
            width: float=None,
            ) -> go.Figure:
            if data.empty:
                raise Exception(f"Provided empty pd.DataFrame: {data}")
            # Reduce the data to the selction
            data = data[variable_list]

            # Format values as strings with 2 decimal places
            if show_data_values:
                text_values = data.round(round_values).astype(str).values
            else:
                text_values = []

            fig = go.Figure(
                data=go.Heatmap(
                    z=data.values,
                    x=data.columns.astype(str),
                    y=data.index.astype(str),
                    zmin=zmin,
                    zmax=zmax,
                    colorscale=colorscale,  # 🍭 rainbow chaos
                    showscale=showscale,
                    hoverongaps=False,
                    colorbar_title=colorbar_title,
                    text=text_values,
                    texttemplate="%{text}",  
                    textfont={
                        "size": text_font_size,
                        "color": "black"}  
                )
            )

            fig.update_layout(
                title=title,
                # font=dict(
                #     family="Papyrus",  # 
                #     size=20,
                #     color="darkred"
                # ),
                xaxis=dict(
                    tickangle=0,
                    title=xaxis_title,
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    title=yaxis_title,
                    showgrid=True,
                    zeroline=False
                ),
                # plot_bgcolor="peachpuff",
                # paper_bgcolor="lavenderblush",
                height=height,
                width=width
            )

            html_fig = to_html(fig)
            print("FUNCTION: get_heatmap_chart")
            if save_graph:
                if file_name is not None:
                    # either take the given path of the function or use the default specified one
                    if file_path is not None:
                        # use the user specified in the function
                        graph_full_save_path = rf'{file_path}/{file_name}'
                    else:
                        graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
            if return_html_version:
                return fig, html_fig
            else:
                return fig


    def lighten_color(
            self,
            hex_color: str,
            factor: float
            ) -> str:
            """Lightens the given color by multiplying it with the given factor (0-1)."""
            rgb = np.array(mcolors.to_rgb(hex_color))
            light_rgb = 1 - (1 - rgb) * factor  # move toward white
            return mcolors.to_hex(light_rgb)


    def create_custom_diverging_colorscale(
            self,
            start_hex: str,
            end_hex: str,
            steps: int = 5,
            center_color: str = "#ffffff",
            lightening_factor: float = 0.6
            ) -> List[[float, str]]:
            """
            Returns a Plotly-style custom diverging continuous color scale from start to end color.
            
            Parameters:
            - start_hex: Hex color for the low end (e.g., '#ff0000')
            - end_hex: Hex color for the high end (e.g., '#0000ff')
            - steps: Number of gradient steps toward the center for each side
            - lightening_factor: Value < 1 to determine how much each step lightens
            
            Returns:
            - List of [position, color] for use in Plotly
            """

            # center_color = "#ffffff"  # divergence midpoint (white)

            # Generate lightened steps from start -> white
            left_colors = [
                self.lighten_color(start_hex, lightening_factor ** (steps - i - 1))
                for i in range(steps)
            ][::-1]

            # Generate lightened steps from end -> white
            right_colors = [
                self.lighten_color(end_hex, lightening_factor ** i)
                for i in range(steps)
            ][::-1]

            # Combine with positions from 0 to 1
            total = 2 * steps + 1
            full_colors = left_colors + [center_color] + right_colors
            scale = [[i / (total - 1), c] for i, c in enumerate(full_colors)]

            return scale


    def get_monthly_yearly_heatmap_chart(
            self,
            data: pd.DataFrame,
            variable_list: List[str],
            title: str="",
            xaxis_title: str="",
            yaxis_title: str="",
            save_graph: bool=False,
            file_name: str=None,
            file_path: str=None,
            return_html_version: bool=False,
            colorscale: str="PiYG",
            colorbar_title: str="Return",
            zmin: float=None,
            zmax: float=None,
            height: float=None,
            width: float=None,
            ) -> go.Figure:
            if data.empty:
                raise Exception(f"Provided empty pd.DataFrame: {data}")
            # Reduce the data to the selction
            data = data[variable_list]

            heatmap_data = {}
            for ticker in data.columns:
                pivot_table = data[ticker].unstack(level=0)  # Years as columns, months as rows
                heatmap_data[ticker] = pivot_table

            fig = make_subplots(
                rows=1,
                cols=len(heatmap_data),
                subplot_titles=list(heatmap_data.keys()),
                shared_yaxes=True,
                shared_xaxes=False
            )
            all_data_values = data.values.flatten()
            # zmin = all_data_values.min()
            # zmax = all_data_values.max()
            sorted_values = np.sort(all_data_values)  # Sort all data values
            # zmin = sorted_values[1]  # Second smallest value
            # zmax = sorted_values[-2]

            for i, (ticker, data) in enumerate(heatmap_data.items(), start=1):
                ticker_heatmap_figure = self.get_heatmap_chart(
                        data=data,
                        variable_list=data.columns,
                        colorscale=colorscale,
                        colorbar_title=colorbar_title,
                        zmin=zmin,
                        zmax=zmax,
                        return_html_version=False
                    )
                ticker_heatmap = ticker_heatmap_figure.data[0]
                fig.add_trace(ticker_heatmap, row=1, col=i)

            fig.update_layout(
                title=title,
                # xaxis_title=xaxis_title,
                # yaxis_title=yaxis_title,
                height=height,
                width=width,  # Adjust width based on the number of tickers
                xaxis=dict(title=xaxis_title),  # Set overall x-axis title
                yaxis=dict(title=yaxis_title)
            )
            html_fig = to_html(fig)
            print("FUNCTION: get_monthly_yearly_heatmap_chart")
            if save_graph:
                if file_name is not None:
                    # either take the given path of the function or use the default specified one
                    if file_path is not None:
                        # use the user specified in the function
                        graph_full_save_path = rf'{file_path}/{file_name}'
                    else:
                        graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
            if return_html_version:
                return fig, html_fig
            else:
                return fig


    def get_bar_chart(
            self,
            data: pd.DataFrame,
            variable_list: List[str],
            title: str="",
            xaxis: str=None,
            xaxis_title: str="Date",
            yaxis_title: str="Value",
            color_discrete_sequence: List[str]=None,
            save_graph: bool=False,
            file_name: str=None,
            file_path: str=None,
            return_html_version: bool=False,
            showlegend: bool=True,
            opacity: float=1.0,
            legend_title: str="",
            height: float=None,
            width: float=None,
            ) -> go.Figure:
            if data.empty:
                raise Exception(f"Provided empty pd.DataFrame: {data}")
            # Reduce the data to the selction
            data = data[variable_list]

            # Set the default color sequence as variable to use
            if color_discrete_sequence is None:
                color_discrete_sequence = self.color_discrete_sequence_default

            # Extract the x-axis data
            if (xaxis is not None and xaxis in data.columns):
                x = data[xaxis].astype(str)
                # since we loop through the columns of the data, make sure to exclude the
                # respective column
                data = data.drop(xaxis, axis=1)
            elif (xaxis not in data.columns):
                print(f"Chosen xaxis variable: {xaxis} not present in data.columns: {data.columns}")
                x = data.index.astype(str)
            else:
                x = data.index.astype(str)
            
            fig = go.Figure()
            for i, var in enumerate(variable_list):
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=data[var],
                        name=str(var),
                        showlegend=showlegend,
                        opacity=opacity,
                        marker_color=color_discrete_sequence[i % len(color_discrete_sequence)]
                    )
                )

            fig.update_layout(
                barmode='group',  # side-by-side bars
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                title=title,
                legend_title=dict(text=legend_title),
                height=height,
                width=width
            )

            html_fig = to_html(fig)
            print("FUNCTION: get_bar_chart")
            if save_graph:
                if file_name is not None:
                    # either take the given path of the function or use the default specified one
                    if file_path is not None:
                        # use the user specified in the function
                        graph_full_save_path = rf'{file_path}/{file_name}'
                    else:
                        graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
            if return_html_version:
                return fig, html_fig
            else:
                return fig


    def get_return_histogram(
            self,
            data: pd.DataFrame,
            variable_list: List[str],
            title: str="",
            xaxis_title: str="",
            yaxis_title: str="",
            color_discrete_sequence: List[str]=None,
            save_graph: bool=False,
            file_name: str=None,
            file_path: str=None,
            return_html_version: bool=False,
            showlegend: bool=True,
            opacity: float=1.0,
            legend_title: str="",
            height: float=None,
            width: float=None,
            draw_vertical_line_at_0: bool=True,
            draw_normal_distribution: bool=True
            ) -> go.Figure:
            if data.empty:
                raise Exception(f"Provided empty pd.DataFrame: {data}")
            # Reduce the data to the selction
            data = data[variable_list]

            # Set the default color sequence as variable to use
            if color_discrete_sequence is None:
                color_discrete_sequence = self.color_discrete_sequence_default

            # create a big plot with the amout of subplots
            mode = ""
            hist_fig = make_subplots(
                 rows=1,
                 cols=len(data.columns)
                 )
            for i, var in enumerate(variable_list):
                histogram = go.Histogram(
                    x=data[var].values,
                    name=str(var),
                    marker_color=color_discrete_sequence[i % len(color_discrete_sequence)],
                    histnorm=mode)

                hist_fig.add_trace(
                    histogram,
                        row=1,
                        col=i+1
                        )
                if draw_normal_distribution:
                    # Add the normal distribution to plot against it
                    x_vals = np.linspace(
                         start=min(data[var].values),
                         stop=max(data[var].values),
                         num=10000
                         )
                    norm_pdf = stats.norm.pdf(
                         x=x_vals,
                         loc=np.mean(data[var].values),
                         scale=np.std(data[var].values)
                         )
                    hist_fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=norm_pdf,
                            name=f"{var}_norm",
                            marker_color="black",
                            mode="lines",
                            showlegend=showlegend,
                            opacity=opacity
                            ),
                            row=1,
                            col=i+1
                        )
                # Add a vertical solid line at the x-value: 0
                counts, bin_edges = np.histogram(data[var].values, bins="auto")
                y_max = max(counts)  # Maximum value of the normal distribution curve

                # Add a vertical solid line at x=0 spanning the inferred y-axis height
                if draw_vertical_line_at_0:
                    hist_fig.add_trace(
                        go.Scatter(
                            x=[0, 0],  # Vertical line at x=0
                            y=[0, y_max],  # Extend from y=0 to max_y
                            name=f"{var}_vertical",
                            marker_color="red",
                            mode="lines",
                            showlegend=False
                            ),
                        row=1,
                        col=i + 1
                    )

            hist_fig.update_layout(
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                title=title,
                legend_title=dict(text=legend_title),
                height=height,
                width=width,
                showlegend=True)
            html_fig = to_html(hist_fig)
            print("FUNCTION: get_return_histogram")
            if save_graph:
                if file_name is not None:
                    # either take the given path of the function or use the default specified one
                    if file_path is not None:
                        # use the user specified in the function
                        graph_full_save_path = rf'{file_path}/{file_name}'
                    else:
                        graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                hist_fig.write_html(graph_full_save_path)
            if return_html_version:
                return hist_fig, html_fig
            else:
                return hist_fig


    def get_scatter_chart(
            self,
            data: pd.DataFrame,
            yaxis: str=None,
            color_discrete_sequence: List[str]=None,
            title: str="",
            xaxis: str=None,
            xaxis_title: str="Date",
            yaxis_title: str="Value",
            save_graph: bool=False,
            file_name: str=None,
            file_path: str=None,
            return_html_version: bool=False,
            showlegend: bool=True,
            opacity: float=1.0,
            legend_title: str="",
            height: float=None,
            width: float=None,
            mode: str='markers',
            ) -> go.Figure:
            if data.empty:
                raise Exception(f"Provided empty pd.DataFrame: {data}")
            
            # Set the default color sequence as variable to use
            if color_discrete_sequence is None:
                color_discrete_sequence = self.color_discrete_sequence_default

            fig = go.Figure()
            for i, index_name in enumerate(data.index):
                fig.add_trace(
                    go.Scatter(
                        x=[data.loc[index_name, xaxis]],  # x-axis: Std. Dev.
                        y=[data.loc[index_name, yaxis]],    # y-axis: Returns
                        mode=mode,
                        name=index_name,  # Legend name: index value
                        opacity=opacity,
                        showlegend=showlegend,
                        marker=dict(
                            color=color_discrete_sequence[i % len(color_discrete_sequence)],
                            size=8.0
                        )
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                legend_title=dict(text=legend_title),
                height=height,
                width=width
            )

            html_fig = to_html(fig)
            print("FUNCTION: get_scatter_chart")
            if save_graph:
                if file_name is not None:
                    # either take the given path of the function or use the default specified one
                    if file_path is not None:
                        # use the user specified in the function
                        graph_full_save_path = rf'{file_path}/{file_name}'
                    else:
                        graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
            if return_html_version:
                return fig, html_fig
            else:
                return fig


    def get_acf_plot(
        self,
        data: pd.Series,
        lags: int = 40,
        title: str = "Autocorrelation Function (ACF)",
        xaxis_title: str = "Lag",
        yaxis_title: str = "ACF Value",
        color: str = "#9b0a7d",
        show_confidence_bands: bool = False,  # New parameter for confidence bands
        alpha: float = 0.05,  # Significance level for confidence bands
        save_graph: bool = False,
        file_name: str = None,
        file_path: str = None,
        return_html_version: bool = False,
        height: float = None,
        width: float = None
        ) -> go.Figure:
        """
        Creates a customizable ACF plot using Plotly with optional confidence bands.

        Args:
            data (pd.Series): Time series data for ACF calculation.
            lags (int): Number of lags to calculate ACF for.
            title (str): Title of the plot.
            xaxis_title (str): Title for the x-axis.
            yaxis_title (str): Title for the y-axis.
            color (str): Color for the lines and markers in the plot.
            show_confidence_bands (bool): Whether to show confidence bands.
            alpha (float): Significance level for confidence bands.
            save_graph (bool): Whether to save the graph as an HTML file.
            file_name (str): Name of the file to save the graph.
            file_path (str): Path to save the graph.
            return_html_version (bool): Whether to return the HTML version of the graph.
            height (float): Height of the plot.
            width (float): Width of the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        if data.empty:
            raise Exception(f"Provided empty pd.Series: {data}")

        # Calculate ACF values
        acf_values, confint = acf(data, nlags=lags, fft=True, alpha=alpha)

        # Create the ACF plot
        fig = go.Figure()

        # Add slim bars starting from y=0 to the ACF value
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                marker=dict(color=color),
                width=0.1,  # Slim bar width
                name="ACF Bars",
                showlegend=False
            )
        )

        # Add thick dots at the tip of each bar
        fig.add_trace(
            go.Scatter(
                x=list(range(len(acf_values))),
                y=acf_values,
                mode='markers',  # Only markers (dots)
                marker=dict(color=color, size=10),  # Customize marker size and color
                name="ACF Dots"
            )
        )

        # Add confidence bands if enabled
        if show_confidence_bands:
            lower_bound = confint[:, 0]
            upper_bound = confint[:, 1]

            # Ensure the x and y values are constructed correctly
            x_values = list(range(len(acf_values))) + list(range(len(acf_values) - 1, -1, -1))  # Forward and reverse indices
            y_values = list(upper_bound) + list(lower_bound[::-1])  # Upper bounds followed by reversed lower bounds

            # Add shaded area for confidence bands
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.1)',  # Light blue shading
                    line=dict(color='rgba(255,255,255,0)'),  # No border line
                    name="Confidence Bands"
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            width=width
        )

        html_fig = to_html(fig)
        print("FUNCTION: get_acf_plot")
        if save_graph:
            if file_name is not None:
                # either take the given path of the function or use the default specified one
                if file_path is not None:
                    graph_full_save_path = rf'{file_path}/{file_name}'
                else:
                    graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
        if return_html_version:
            return fig, html_fig
        else:
            return fig

    def get_pacf_plot(
        self,
        data: pd.Series,
        lags: int = 40,
        title: str = "Partial Autocorrelation Function (PACF)",
        xaxis_title: str = "Lag",
        yaxis_title: str = "PACF Value",
        color: str = "#39842e",
        show_confidence_bands: bool = False,  # New parameter for confidence bands
        alpha: float = 0.05,  # Significance level for confidence bands
        save_graph: bool = False,
        file_name: str = None,
        file_path: str = None,
        return_html_version: bool = False,
        height: float = None,
        width: float = None
        ) -> go.Figure:
        """
        Creates a customizable PACF plot using Plotly with optional confidence bands.

        Args:
            data (pd.Series): Time series data for PACF calculation.
            lags (int): Number of lags to calculate PACF for.
            title (str): Title of the plot.
            xaxis_title (str): Title for the x-axis.
            yaxis_title (str): Title for the y-axis.
            color (str): Color for the lines and markers in the plot.
            show_confidence_bands (bool): Whether to show confidence bands.
            alpha (float): Significance level for confidence bands.
            save_graph (bool): Whether to save the graph as an HTML file.
            file_name (str): Name of the file to save the graph.
            file_path (str): Path to save the graph.
            return_html_version (bool): Whether to return the HTML version of the graph.
            height (float): Height of the plot.
            width (float): Width of the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        if data.empty:
            raise Exception(f"Provided empty pd.Series: {data}")

        # Calculate PACF values
        pacf_values, confint = pacf(data, nlags=lags, alpha=alpha)

        # Create the PACF plot
        fig = go.Figure()

        # Add slim bars starting from y=0 to the PACF value
        fig.add_trace(
            go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                marker=dict(color=color),
                width=0.1,  # Slim bar width
                name="PACF Bars",
                showlegend=False
            )
        )

        # Add thick dots at the tip of each bar
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                mode='markers',  # Only markers (dots)
                marker=dict(color=color, size=10),  # Customize marker size and color
                name="PACF Dots"
            )
        )

        # Add confidence bands if enabled
        if show_confidence_bands:
            lower_bound = confint[:, 0]
            upper_bound = confint[:, 1]

            # Add shaded area for confidence bands
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pacf_values))) + list(range(len(pacf_values))[::-1]),
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),  # No border line
                    name="Confidence Bands"
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            width=width
        )

        html_fig = to_html(fig)
        print("FUNCTION: get_pacf_plot")
        if save_graph:
            if file_name is not None:
                # either take the given path of the function or use the default specified one
                if file_path is not None:
                    graph_full_save_path = rf'{file_path}/{file_name}'
                else:
                    graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                # save the figure
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
        if return_html_version:
            return fig, html_fig
        else:
            return fig


    def get_frequency_distribution_plot(
        self,
        data: pd.Series,
        frequency: str = "monthly",  # Options: "weekly", "monthly", "quarterly", "yearly"
        nbins: int = 30,
        title: str = "Frequency Distribution Plot",
        xaxis_title: str = "Frequency",
        yaxis_title: str = "Value",
        color_discrete_sequence: List[str] = None,
        save_graph: bool = False,
        file_name: str = None,
        file_path: str = None,
        return_html_version: bool = False,
        height: float = None,
        width: float = None
        ) -> go.Figure:
        """
        Creates a frequency distribution plot using Plotly.

        Args:
            data (pd.Series): Time series data for distribution analysis.
            frequency (str): Frequency for grouping data ("weekly", "monthly", "quarterly", "yearly").
            nbins (int): Number of bins for histogram.
            title (str): Title of the plot.
            xaxis_title (str): Title for the x-axis.
            yaxis_title (str): Title for the y-axis.
            color_discrete_sequence (List[str]): List of colors for the plot.
            save_graph (bool): Whether to save the graph as an HTML file.
            file_name (str): Name of the file to save the graph.
            file_path (str): Path to save the graph.
            return_html_version (bool): Whether to return the HTML version of the graph.
            height (float): Height of the plot.
            width (float): Width of the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        if data.empty:
            raise Exception(f"Provided empty pd.Series: {data}")

        # Group data by the specified frequency
        # if frequency == "weekly":
        #    grouped_data = data.groupby(data.index.week)
        elif frequency == "monthly":
            grouped_data = data.groupby(data.index.month)
        elif frequency == "quarterly":
            grouped_data = data.groupby(data.index.quarter)
        elif frequency == "yearly":
            grouped_data = data.groupby(data.index.year)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        fig = go.Figure()

        if frequency == "yearly":
            # Prepare data for distplot
            hist_data = [values for year, values in grouped_data]
            group_labels = [str(year) for year in grouped_data.groups.keys()]

            # Create distplot with custom bin size
            fig = create_distplot(
                hist_data=hist_data,
                group_labels=group_labels,
                histnorm='probability',
                # bin_size=0.05,  # Adjust bin size for finer/coarser distribution
                show_rug=True
            )

            # # Overlaid histograms for yearly frequency
            # for year, values in grouped_data:
            #     fig.add_trace(
            #         go.Histogram(
            #             x=values,
            #             name=str(year),
            #             opacity=0.5,  # Semi-transparent for overlay
            #             histnorm='probability density',
            #             nbinsx=nbins,
            #             marker_color=color_discrete_sequence[list(grouped_data.groups.keys()).index(year) % len(color_discrete_sequence)]
            #         )
            #     )
        else:
            # Violin plots for higher frequencies
            for freq, values in grouped_data:
                fig.add_trace(
                    go.Violin(
                        x=[freq] * len(values),  # Frequency as x-axis
                        y=values,
                        name=str(freq),
                        box_visible=True,  # Show box plot inside violin
                        meanline_visible=True,  # Show mean line
                        marker_color=color_discrete_sequence[list(grouped_data.groups.keys()).index(freq) % len(color_discrete_sequence)]
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            width=width
        )

        html_fig = to_html(fig)
        print("FUNCTION: get_frequency_distribution_plot")
        if save_graph:
            if file_name is not None:
                if file_path is not None:
                    graph_full_save_path = rf'{file_path}/{file_name}'
                else:
                    graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
        if return_html_version:
            return fig, html_fig
        else:
            return fig


    def get_seasonal_decomposition_plot(
        self,
        decomposition_result,
        title: str = "Seasonal Decomposition",
        save_graph: bool = False,
        file_name: str = None,
        file_path: str = None,
        return_html_version: bool = False,
        height: float = None,
        width: float = None,
        color_discrete_sequence: List[str] = None
        ) -> go.Figure:
        """
        Creates a Plotly visualization for the four components of seasonal decomposition.

        Args:
            decomposition_result: Decomposition result from statsmodels.tsa.seasonal_decompose.
            title (str): Title of the plot.
            save_graph (bool): Whether to save the graph as an HTML file.
            file_name (str): Name of the file to save the graph.
            file_path (str): Path to save the graph.
            return_html_version (bool): Whether to return the HTML version of the graph.
            height (float): Height of the plot.
            width (float): Width of the plot.

        Returns:
            go.Figure: Plotly figure object.
        """
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"]
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )

        # Add observed data
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.observed.index,
                y=decomposition_result.observed,
                mode='lines',
                name='Observed',
                line=dict(color=color_discrete_sequence[0] if color_discrete_sequence else "#1f77b4")  # Default blue
            ),
            row=1,
            col=1
        )

        # Add trend component
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.trend.index,
                y=decomposition_result.trend,
                mode='lines',
                name='Trend',
                line=dict(color=color_discrete_sequence[1] if color_discrete_sequence else "#ff7f0e")  # Default orange
            ),
            row=2,
            col=1
        )

        # Add seasonal component
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.seasonal.index,
                y=decomposition_result.seasonal,
                mode='lines',
                name='Seasonal',
                line=dict(color=color_discrete_sequence[2] if color_discrete_sequence else "#2ca02c")  # Default green
            ),
            row=3,
            col=1
        )

        # Add residual component
        fig.add_trace(
            go.Scatter(
                x=decomposition_result.resid.index,
                y=decomposition_result.resid,
                mode='lines',
                name='Residual',
                line=dict(color=color_discrete_sequence[3] if color_discrete_sequence else "#d62728")  # Default red
            ),
            row=4,
            col=1
        )

        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            width=width,
            xaxis_title="Date",
            yaxis_title="Value"
        )

        html_fig = to_html(fig)
        print("FUNCTION: get_seasonal_decomposition_plot")
        if save_graph:
            if file_name is not None:
                if file_path is not None:
                    graph_full_save_path = rf'{file_path}/{file_name}'
                else:
                    graph_full_save_path = rf'{self.project_path}/{self.results_path_ending}/{self.graphs_path_ending}/{file_name}'
                path_till_folder_name = "/".join(graph_full_save_path.split("/")[:-1])
                self.__check_path_existence(path=path_till_folder_name)
                fig.write_html(graph_full_save_path)
        if return_html_version:
            return fig, html_fig
        else:
            return fig


# Example usage
# PROJECT_PATH = r""
# RESULTS_PATH_ENDING = r"results/"
# GRAPHS_PATH_ENDING = r"graphs/"
# CAU_COLOR_SCALE = ["#9b0a7d", "grey", "black", "darkgrey", "lightgrey"]
# COLOR_DISCRETE_SEQUENCE_DEFAULT = CAU_COLOR_SCALE

# standard_plotting_instance = StandardPlotting(
#     project_path=PROJECT_PATH,
#     results_path_ending=RESULTS_PATH_ENDING,
#     graphs_path_ending=GRAPHS_PATH_ENDING,
#     color_discrete_sequence_default=COLOR_DISCRETE_SEQUENCE_DEFAULT
#     )
# # Basic Time-series analysis
# os.chdir(r"/Users/Robert_Hennings/Projects/STANDARD_STRUCTURES/standard_structures")
# credential_path = r"/Users/Robert_Hennings/SettingsPackages"
# credential_file_name = "credentials.json"

# from standard_data_loading import StandardDataLoading
# standard_data_loading_instance = StandardDataLoading(
#     credential_path=credential_path,
#     credential_file_name=credential_file_name
#     )

# ticker = ["^GDAXI", "^DJI", "^N225", "^GSPC", "^NYA", "^N100"]
# start_date = (datetime.today() - pd.Timedelta(days=365*5))
# end_date = datetime.today()
# test_data = standard_data_loading_instance.get_yahoo_data(
#     ticker=ticker,
#     start_date=start_date,
#     end_date=end_date
#     )
# filter_columns = ["Adj Close"]
# test_data_filtered = standard_data_loading_instance.filter_yahoo_data(
#     yahoo_df=test_data,
#     columns=filter_columns
#     )
# test_data_filtered.columns = test_data_filtered.columns.droplevel(level=0)  # Remove the first level of MultiIndex
# # periods=3000
# # test_data_filtered = pd.DataFrame(
# #     index=pd.date_range(start="2020-01-01", periods=periods, freq='D'),
# #     data={
# #         'AAPL': np.cumsum(np.random.rand(periods)),
# #         'GOOGL': np.cumsum(np.random.rand(periods)),
# #         'MSFT': np.cumsum(np.random.rand(periods)),
# #         'AMZN': np.cumsum(np.random.rand(periods))
# #     })
# # Create a basic plotly line chart, plotting all columns as separate lines
# # Define a custom diverging color scale
# custom_color_scale = standard_plotting_instance.create_custom_diverging_colorscale(
#     start_hex="#9b0a7d",
#     end_hex="black",
#     center_color="grey",
#     steps=round(len(test_data_filtered.columns)/2),
#     lightening_factor=0.8,
# )
# # Extract only the hex color codes from the created list
# custom_color_scale_codes = [color[1] for color in custom_color_scale]

# standard_plotting_line_chart = standard_plotting_instance.get_line_chart(
#     data=test_data_filtered,
#     variable_list=test_data_filtered.columns,
#     color_discrete_sequence=custom_color_scale_codes,
#     title=f"Adj Close over the time range: {test_data_filtered.index[0].strftime('%d-%m-%Y')} - {test_data_filtered.index[-1].strftime('%d-%m-%Y')}",
#     xaxis_title="Date",
#     yaxis_title="Adj Close",
#     legend_title="Stock Ticker"
# )
# standard_plotting_line_chart.show(renderer="browser")
# # Plot data using a secondary y-axis
# dual_axis_chart = standard_plotting_instance.get_dual_axis_line_chart(
#     data=test_data_filtered,
#     variable_list=test_data_filtered.columns,
#     secondary_yaxis_variables=["^GSPC", "^DJI"],  # Variables for the secondary y-axis
#     color_discrete_sequence=custom_color_scale_codes,
#     title="Dual Axis Line Chart",
#     xaxis_title="Date",
#     yaxis_title="Primary Y-Axis",
#     secondary_yaxis_title="Secondary Y-Axis",
#     legend_title="Stock Ticker"
# )
# dual_axis_chart.show(renderer="browser")
# # 1) ACF Plot
# acf_plot = standard_plotting_instance.get_acf_plot(
#     data=test_data_filtered['^GDAXI'].dropna(),
#     lags=30,
#     title="ACF Plot for ^GDAXI",
#     xaxis_title="Lag",
#     show_confidence_bands=True,
#     yaxis_title="ACF Value",
#     color="#9b0a7d",
#     save_graph=False,
#     file_name="acf_plot.html"
# )
# acf_plot.show(renderer="browser")
# # 2) PACF Plot
# pacf_plot = standard_plotting_instance.get_pacf_plot(
#     data=test_data_filtered['^GDAXI'].dropna(),  # Replace with your time series data
#     lags=30,
#     title="PACF Plot for ^GDAXI",
#     xaxis_title="Lag",
#     yaxis_title="PACF Value",
#     color="#9b0a7d",
#     show_confidence_bands=True,  # Enable confidence bands
#     alpha=0.05,  # 95% confidence interval
#     save_graph=False,
#     file_name="pacf_plot.html"
# )
# pacf_plot.show(renderer="browser")
# # 3) Observe the distribution of the data for a given frequency, i.e. weekly, monthly, quarterly, yearly as a histogram
# # Yearly as just overlaid histograms, all higher frequencies as violion plots
# # Example usage
# frequency_plot = standard_plotting_instance.get_frequency_distribution_plot(
#     data=test_data_filtered['^GDAXI'].pct_change().dropna()*100,  # Replace with your time series data
#     frequency="yearly",  # Options: "monthly", "quarterly", "yearly"
#     title="Monthly Frequency Distribution for ^GDAXI",
#     xaxis_title="Month",
#     yaxis_title="Value",
#     color_discrete_sequence=["#9b0a7d", "#39842e", "#2e3984", "#7d9b0a"],
#     save_graph=False,
#     file_name="monthly_distribution_plot.html"
# )
# frequency_plot.show(renderer="browser")
# # 4) Time-series decomposition
# series = test_data_filtered['^GDAXI']
# series = series.reindex(pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')).fillna(method='ffill')
# model = 'additive'  # or 'multiplicative'
# result = seasonal_decompose(series, model=model)

# decomposition_plot = standard_plotting_instance.get_seasonal_decomposition_plot(
#     decomposition_result=result,
#     title=f"Seasonal Decomposition of AAPL, assuming an {model}-model",
#     save_graph=False,
#     file_name="seasonal_decomposition_plot.html",
#     color_discrete_sequence=CAU_COLOR_SCALE
# )
# decomposition_plot.show(renderer="browser")

# model = 'multiplicative'  # or 'multiplicative'
# result = seasonal_decompose(series, model=model)
# decomposition_plot = standard_plotting_instance.get_seasonal_decomposition_plot(
#     decomposition_result=result,
#     title=f"Seasonal Decomposition of AAPL, assuming an {model}-model",
#     save_graph=False,
#     file_name="seasonal_decomposition_plot.html",
#     color_discrete_sequence=CAU_COLOR_SCALE
# )
# decomposition_plot.show(renderer="browser")
# # From the decomposition according to the two model variants, we can conclude that there are 
# # time-varying Time-series features that have to be modelled and considered

# # 5) Boxplots per Day of the Week, Week, Month, Quarter
# def create_yearly_boxplot_subplots(
#     data: pd.DataFrame,
#     variable: str,
#     time_axis: str = "Month",  # Options: "Day of the Week", "Week", "Month", "Quarter"
#     title: str = "Yearly Boxplots",
#     height_per_subplot: int = None,
#     width: int = None,
#     box_color: str = "grey",  # Color of the box
#     line_color: str = "grey",  # Color of the outer lines
#     mean_color: str = "grey",  # Color of the mean line
#     dot_color: str = "#9b0a7d"  # Color of the single dots
# ) -> go.Figure:
#     """
#     Creates subplots for each unique year in the data, where each subplot contains boxplots
#     of the time series variable based on the specified time axis.

#     Args:
#         data (pd.DataFrame): Time series data with a DateTime index.
#         variable (str): The column name of the variable to plot.
#         time_axis (str): The x-axis grouping ("Day of the Week", "Week", "Month", "Quarter").
#         title (str): Title of the entire figure.
#         height_per_subplot (int): Height of each subplot.
#         width (int): Width of the entire figure.
#         box_color (str): Color of the box.
#         line_color (str): Color of the outer lines.
#         mean_color (str): Color of the mean line.
#         dot_color (str): Color of the single dots.

#     Returns:
#         go.Figure: Plotly figure object with subplots.
#     """
#     if data.empty:
#         raise ValueError("The provided DataFrame is empty.")
#     if variable not in data.columns:
#         raise ValueError(f"The variable '{variable}' is not in the DataFrame columns.")

#     # Map time_axis to pandas attributes
#     time_axis_mapping = {
#         "Day of the Week": data.index.dayofweek,
#         "Week": data.index.isocalendar().week,
#         "Month": data.index.month,
#         "Quarter": data.index.quarter
#     }
#     if time_axis not in time_axis_mapping:
#         raise ValueError(f"Invalid time_axis '{time_axis}'. Choose from: {list(time_axis_mapping.keys())}.")

#     # Add a column for the time axis
#     data["TimeAxis"] = time_axis_mapping[time_axis]
#     data["Year"] = data.index.year

#     # Get unique years
#     unique_years = sorted(data["Year"].unique())

#     # Create subplots
#     fig = make_subplots(
#         rows=len(unique_years),
#         cols=1,
#         shared_xaxes=True,
#         subplot_titles=[f"Year: {year}" for year in unique_years]
#     )

#     # Add boxplots for each year
#     for i, year in enumerate(unique_years, start=1):
#         yearly_data = data[data["Year"] == year]
#         fig.add_trace(
#             go.Box(
#                 x=yearly_data["TimeAxis"],
#                 y=yearly_data[variable],
#                 name=str(year),
#                 boxmean=True,  # Show mean line
#                 marker=dict(
#                     color=box_color,  # Color of the box
#                     outliercolor=dot_color,  # Color of the single dots
#                     line=dict(color=line_color)  # Color of the outer lines
#                 ),
#                 line=dict(color=mean_color)  # Color of the mean line
#             ),
#             row=i,
#             col=1
#         )

#     # Define x-tick labels based on the time_axis
#     tick_labels_mapping = {
#         "Day of the Week": {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"},
#         "Week": {i: str(i) for i in range(1, 53)},  # Weeks 1 to 52
#         "Month": {i: month for i, month in enumerate(
#             ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)},
#         "Quarter": {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
#     }

#     # Get the tick values and labels for the selected time_axis
#     tick_vals = list(tick_labels_mapping[time_axis].keys())
#     tick_labels = list(tick_labels_mapping[time_axis].values())

#     # Update layout
#     fig.update_layout(
#         title=title,
#         # height=height_per_subplot * len(unique_years),
#         width=width,
#         xaxis=dict(
#             title=time_axis,
#             tickmode="array",
#             tickvals=tick_vals,
#             ticktext=tick_labels
#         ),
#         yaxis_title=variable,
#         showlegend=False
#     )
#     return fig

# # Example usage
# boxplot_figure = create_yearly_boxplot_subplots(
#     data=test_data_filtered,
#     variable="^GDAXI",
#     time_axis="Month",  # Options: "Day of the Week", "Week", "Month", "Quarter"
#     title="Yearly Boxplots of ^GDAXI by Month"
# )
# boxplot_figure.show(renderer="browser")

# # 6) The Log Time-series and its growth in log first differences
# def plot_time_series_with_growth(
#     data: pd.Series,
#     title: str = "Time Series and Growth",
#     yaxis_title: str = "Log Values",
#     secondary_yaxis_title: str = "Log First Differences (Growth)",
#     log_color: str = "blue",
#     growth_color: str = "red",
#     height: int = None,
#     width: int = None
# ) -> go.Figure:
#     """
#     Plots a time series in logs on the primary y-axis and its log first differences on a secondary y-axis.

#     Args:
#         data (pd.Series): Time series data with a DateTime index.
#         title (str): Title of the plot.
#         yaxis_title (str): Title for the primary y-axis.
#         secondary_yaxis_title (str): Title for the secondary y-axis.
#         log_color (str): Color for the log-transformed time series line.
#         growth_color (str): Color for the growth (log first differences) line.
#         height (int): Height of the plot.
#         width (int): Width of the plot.

#     Returns:
#         go.Figure: Plotly figure object.
#     """
#     if data.empty:
#         raise ValueError("The provided Series is empty.")

#     # Calculate log-transformed values and log first differences
#     log_values = np.log(data)
#     log_diff = log_values.diff().dropna()

#     # Create the figure
#     fig = go.Figure()

#     # Add the log-transformed time series to the primary y-axis
#     fig.add_trace(
#         go.Scatter(
#             x=log_values.index,
#             y=log_values,
#             mode="lines",
#             name="Log Values",
#             line=dict(color=log_color, width=2)
#         )
#     )

#     # Add the log first differences (growth) to the secondary y-axis
#     fig.add_trace(
#         go.Scatter(
#             x=log_diff.index,
#             y=log_diff,
#             mode="lines",
#             name="Log First Differences (Growth)",
#             line=dict(color=growth_color, width=2),
#             yaxis="y2"  # Assign to secondary y-axis
#         )
#     )

#     # Update layout to include a secondary y-axis
#     fig.update_layout(
#         title=title,
#         height=height,
#         width=width,
#         xaxis=dict(title="Date"),
#         yaxis=dict(
#             title=yaxis_title,
#             showgrid=True,
#             zeroline=False
#         ),
#         yaxis2=dict(
#             title=secondary_yaxis_title,
#             overlaying="y",
#             side="right",
#             showgrid=False,
#             zeroline=False
#         ),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         )
#     )

#     return fig
# # Example usage
# time_series_growth_plot = plot_time_series_with_growth(
#     data=test_data_filtered['^GDAXI'],
#     title="Log of ^GDAXI and Its Growth",
#     yaxis_title="Log of ^GDAXI",
#     secondary_yaxis_title="Log First Differences (Growth)",
#     log_color="grey",
#     growth_color="#9b0a7d"
# )
# time_series_growth_plot.show(renderer="browser")

# # 7) Plot rolling correlation between two time-series
# def plot_rolling_correlation(
#     data: pd.DataFrame,
#     variable_list: List[str],
#     title: str = "Rolling Correlation",
#     yaxis_title: str = "Correlation",
#     len_rolling_window: int = 30,
#     color_discrete_sequence: List[str]=None,
#     height: int = None,
#     width: int = None
#     ) -> go.Figure:
#     if data.empty:
#         raise ValueError("The provided Series is empty.")

#     fig = go.Figure()
#     for i in range(len(variable_list)):
#             color = color_discrete_sequence[i]
#             var1 = variable_list[i]
#             for j in range(i+1, len(variable_list)):
#                 var2 = variable_list[j]
#                 rolling_corr = data[var1].rolling(window=len_rolling_window).corr(data[var2]).dropna()
#                 fig.add_trace(
#                     go.Scatter(
#                         x=rolling_corr.index,
#                         y=rolling_corr,
#                         mode="lines",
#                         name=f"{var1} & {var2}",
#                         line=dict(color=color, width=2)
#                     )
#                 )
#     fig.update_layout(
#         title=title,
#         yaxis_title=yaxis_title,
#         height=height,
#         width=width,
#         xaxis_title="Date",
#         legend_title="Rolling Correlation Pairs"
#     )
#     return fig
# # Example usage
# rolling_correation_plot = plot_rolling_correlation(
#     data=test_data_filtered,
#     variable_list=['^DJI', '^GDAXI', '^GSPC'],
#     title="30-Day Rolling Correlation between Stock Tickers",
#     yaxis_title="Correlation",
#     len_rolling_window=30,
#     color_discrete_sequence=COLOR_DISCRETE_SEQUENCE_DEFAULT
# )
# rolling_correation_plot.show(renderer="browser")


# # 8) Tests for Stationarity - DF, ADF, Phillips-Perron, Granger Causality, Cointegration
# # Trend Stationary or Difference Stationary
# def adf_test(
#     data: pd.Series,
#     title: str = "Augmented Dickey-Fuller Test",
#     regression_type: str = 'c',  # 'c' for constant, 'ct' for constant and trend, 'nc' for no constant
#     autolag: str = 'AIC',  # 'AIC', 'BIC', 't-stat', None
#     maxlag: int = None,
#     variable: str = None,
#     significance_level: float = 0.05
#     ) -> pd.DataFrame:
#     """
#     Performs the Augmented Dickey-Fuller (ADF) test on a time series and prints the results.

#     Args:
#         data (pd.Series): Time series data to test for stationarity.
#         title (str): Title for the output.
#         regression_type (str): Type of regression ('c', 'ct', 'nc').
#         autolag (str): Method to use for lag selection ('AIC', 'BIC', 't-stat', None).
#         maxlag (int): Maximum number of lags to consider.
#         variable (str): Name of the variable being tested.
#         significance_level (float): Significance level for the test.
#     Returns:
#         pd.DataFrame: DataFrame containing the ADF test results.
#     """
#     if data.empty:
#         raise ValueError("The provided Series is empty.")

#     from statsmodels.tsa.stattools import adfuller
#     data = data[variable].dropna()
#     adf_stat, p_val, crit_vals, result = adfuller(
#         x=data,
#         maxlag=maxlag,
#         regression=regression_type,
#         autolag=autolag,
#         store=True,
#         regresults=False
#         )
#     regression_summary = result.resols.summary()
#     print(f"{title}\n")
#     print(f"ADF Statistic: {adf_stat}")
#     print(f"p-value: {p_val}")
#     print("Critical Values:")
#     for key, value in crit_vals.items():
#         print(f"   {key}: {value}")
#     if result[1] < significance_level:
#         print(f"The null hypothesis can be rejected at the {significance_level*100}% significance level. The series is stationary.")
#     else:
#         print(f"The null hypothesis cannot be rejected at the {significance_level*100}% significance level. The series is non-stationary.")

#     result_df = pd.DataFrame({
#         'ADF Statistic': [adf_stat],
#         'p-value': [p_val],
#         '1% Critical Value': [crit_vals['1%']],
#         '5% Critical Value': [crit_vals['5%']],
#         '10% Critical Value': [crit_vals['10%']],
#         'H0': [result.H0],
#         'HA': [result.HA],
#         'Used Lag:': [result.usedlag],
#         'Max Lag:': [result.maxlag],
#         'Start Time:': [f"{data.index.min().strftime('%d-%m-%Y')}"],
#         'End Time:': [f"{data.index.min().strftime('%d-%m-%Y')}"],
#         'Observations:': [result.nobs],
#         'Variable': [variable],
#         'Tested at': [f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"]
#         })
#     return result_df
# # Example usage
# test_result = adf_test(
#     data=test_data_filtered,
#     variable='^GDAXI',
#     title="ADF Test for ^GDAXI",
#     significance_level=0.05
# )
# test_result = adf_test(
#     data=test_data_filtered['^GDAXI'].diff().dropna(),
#     title="ADF Test for ^GDAXI",
#     significance_level=0.05
# )
# def granger_causality_test(
#     data: pd.DataFrame,
#     variable_x: str,
#     variable_y: str,
#     max_lag: int = 10,
#     significance_level: float = 0.05,
#     test_type_p_value: str = 'ssr_ftest'
#     ) -> None:
#     """
#     Performs the Granger Causality test between two time series and prints the results.

#     Args:
#         data (pd.DataFrame): DataFrame containing the time series data.
#         variable_x (str): The name of the first variable (cause).
#         variable_y (str): The name of the second variable (effect).
#         max_lag (int): Maximum number of lags to test.
#         significance_level (float): Significance level for the test.
#         test_type_p_value (str): Type of test statistic to use for p-value ('ssr_ftest', 'ssr_chi2test', etc.).

#     Returns:
#         None
#     """
#     if data.empty:
#         raise ValueError("The provided DataFrame is empty.")
#     if variable_x not in data.columns or variable_y not in data.columns:
#         raise ValueError(f"One or both variables '{variable_x}', '{variable_y}' are not in the DataFrame columns.")

#     from statsmodels.tsa.stattools import grangercausalitytests
#     print(f"Granger Causality Test between {variable_x} and {variable_y}\n")
#     data = data[[variable_y, variable_x]].dropna()
#     test_result = grangercausalitytests(data, maxlag=max_lag, verbose=True)
#     grenger_test_df_list = []
#     for lag in range(1, max_lag + 1):
#         lag_data = granger_test_result[lag]
#         lag_data = lag_data[0]
#         grenger_test_df = pd.DataFrame()
#         for key in lag_data.keys():
#             lag_diagnostics = pd.DataFrame(
#                 lag_data.get(key),
#                 ).T
#             if lag_diagnostics.shape[1] == 3:
#                 columns = ['Test-Statistic', 'p-value', 'df_num']
#             else:
#                 columns = ['Test-Statistic', 'p-value', 'df_denom', 'df_num']
#             lag_diagnostics.columns = columns
#             lag_diagnostics["Metric"] = key
#             grenger_test_df = pd.concat(
#                 [grenger_test_df,
#                 lag_diagnostics],
#                 axis=0)
#         grenger_test_df["Lag"] = lag
#         grenger_test_df_list.append(grenger_test_df)
#         p_value = test_result[lag][0][test_type_p_value][1]
#         if p_value < significance_level:
#             print(f"Lag {lag}: Reject null hypothesis at {significance_level*100}% significance level. {variable_x} Granger-causes {variable_y}.")
#         else:
#             print(f"Lag {lag}: Cannot reject null hypothesis at {significance_level*100}% significance level. No Granger causality from {variable_x} to {variable_y}.")
#     granger_test_df_complete = pd.concat(grenger_test_df_list, axis=0).reset_index(drop=True)
#     granger_test_df_complete['Start Time'] = f"{data.index.min().strftime('%d-%m-%Y')}"
#     granger_test_df_complete['end Time'] = f"{data.index.max().strftime('%d-%m-%Y')}"
#     granger_test_df_complete['Observations'] = data.shape[0]
#     granger_test_df_complete['Test'] = f"{variable_x} causes {variable_y}"
#     granger_test_df_complete['Tested at'] = f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"

#     return test_result, granger_test_df_complete
# # Example usage
# granger_test_result, granger_test_result_df = granger_causality_test(
#     data=test_data_filtered[['^GDAXI', '^GSPC']],
#     variable_x='^GDAXI',
#     variable_y='^GSPC',
#     max_lag=10,
#     significance_level=0.05
# )
# # Plotting the results
# def plot_granger_test_results(
#     data: pd.DataFrame,
#     title: str = "Granger Causality Test Results",
#     xaxis_title: str = "Lag",
#     yaxis_title: str = "F-Statistic",
#     pvalue_yaxis_title: str = "p-value",
#     significance_level: float = 0.05,
#     height: int = None,
#     width: int = None
#     ) -> go.Figure:
#     # Create a Plotly figure
#     fig = go.Figure()

#     # Plot each Metric as a separate line on the primary Y-axis
#     for metric in data['Metric'].unique():
#         metric_data = data[data['Metric'] == metric]
#         fig.add_trace(
#             go.Scatter(
#                 x=metric_data['Lag'],
#                 y=metric_data['F-Statistic'],
#                 mode='lines+markers',
#                 name=f"F-Statistic ({metric})",
#                 yaxis="y1"  # Assign to primary Y-axis
#             )
#         )

#     # Plot the p-value as a separate line on the secondary Y-axis
#     fig.add_trace(
#         go.Scatter(
#             x=data['Lag'],
#             y=data['p-value'],
#             mode='lines+markers',
#             name="p-value",
#             line=dict(color="red", dash="dash"),
#             yaxis="y2"  # Assign to secondary Y-axis
#         )
#     )

#     # Add a horizontal line for the significance level on the secondary Y-axis
#     fig.add_trace(
#         go.Scatter(
#             x=[data['Lag'].min(), data['Lag'].max()],
#             y=[significance_level, significance_level],
#             mode='lines',
#             name=f"Significance Level ({significance_level})",
#             line=dict(color="green", dash="dot"),
#             yaxis="y2"  # Assign to secondary Y-axis
#         )
#     )

#     # Update layout to include a secondary Y-axis
#     fig.update_layout(
#         title=title,
#         xaxis=dict(title=xaxis_title),
#         yaxis=dict(
#             title=yaxis_title,
#             showgrid=True,
#             zeroline=False
#         ),
#         yaxis2=dict(
#             title=pvalue_yaxis_title,
#             overlaying="y",  # Overlay on the same plot
#             side="right",  # Place on the right side
#             showgrid=False,
#             zeroline=False
#         ),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         height=height,
#         width=width
#     )

#     # Show the plot
#     return fig

# # Example usage
# granger_causality_test_plot = plot_granger_test_results(
#     granger_test_result_df,
#     significance_level=0.05,
#     title="Granger Causality Test Results with p-value",
#     xaxis_title="Lag",
#     yaxis_title="F-Statistic",
#     pvalue_yaxis_title="p-value"
# )
# granger_causality_test_plot.show(renderer="browser")


# def cointegration_test(
#     data: pd.DataFrame,
#     variable_x: str,
#     variable_y: str,
#     significance_level: float = 0.05
#     ) -> pd.DataFrame:
#     """
#     Performs the Engle-Granger cointegration test between two time series and prints the results.

#     Args:
#         data (pd.DataFrame): DataFrame containing the time series data.
#         variable_x (str): The name of the first variable.
#         variable_y (str): The name of the second variable.
#         significance_level (float): Significance level for the test.

#     Returns:
#         None
#     """
#     if data.empty:
#         raise ValueError("The provided DataFrame is empty.")
#     if variable_x not in data.columns or variable_y not in data.columns:
#         raise ValueError(f"One or both variables '{variable_x}', '{variable_y}' are not in the DataFrame columns.")

#     from statsmodels.tsa.stattools import coint
#     data = data[[variable_x, variable_y]].dropna()
#     cointegration_test_result = coint(data[variable_x], data[variable_y])
#     print(f"Cointegration Test between {variable_x} and {variable_y}\n")
#     print(f"Cointegration Score: {cointegration_test_result[0]}")
#     print(f"p-value: {cointegration_test_result[1]}")
#     if cointegration_test_result[1] < significance_level:
#         print(f"The null hypothesis can be rejected at the {significance_level*100}% significance level. The series are cointegrated.")
#     else:
#         print(f"The null hypothesis cannot be rejected at the {significance_level*100}% significance level. The series are not cointegrated.")
#     cointegration_df = pd.DataFrame({
#         'Cointegration Score': [cointegration_test_result[0]],
#         'p-value': [cointegration_test_result[1]],
#         'Start Time': [f"{data.index.min().strftime('%d-%m-%Y')}"],
#         'End Time': [f"{data.index.max().strftime('%d-%m-%Y')}"],
#         'Observations': [data.shape[0]],
#         'Test': [f'{variable_x} and {variable_y}'],
#         'Tested at': [f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"]
#         })
#     return cointegration_df
# # Example usage
# cointegration_test_result_df = cointegration_test(
#     data=test_data_filtered[['^GDAXI', '^GSPC']].diff().dropna(),
#     variable_x='^GDAXI',
#     variable_y='^GSPC',
#     significance_level=0.05
# )



# # Create a basic plotly heatmap, plotting the linear correlation amongst all columns
# test_data_filtered.columns = test_data_filtered.columns.get_level_values(level=1)
# standar_plotting_heatmap_chart = standar_plotting_instance.get_heatmap_chart(
#     data=test_data_filtered.corr(),
#     variable_list=test_data_filtered.columns,
#     title=f"Linear Correlation of the Adj Close over the time range: {start_date.strftime('%d-%m-%Y')} - {end_date.strftime('%d-%m-%Y')}",
#     xaxis_title="Stock Ticker",
#     yaxis_title="Stock Ticker",
#     )
# standar_plotting_heatmap_chart.show(renderer="browser")
# # Create a custom diverging colorscale (for the above mentioned heatmap) ny providing the
# # two outer exrema and the number of lightening steps
# standard_plotting_custom_diverging_colorscale = standar_plotting_instance.create_custom_diverging_colorscale(
#     start_hex="#9b0a7d",
#     end_hex="#39842e",
#     steps=3
#     )
# print(standard_plotting_custom_diverging_colorscale)
# # Create a basic multiple heatmap chart, side by side for each Ticker (column) one heatmap with the Years as x-axis and the months as y-axis
# monthly_grouped = test_data_filtered.groupby([test_data_filtered.index.year, test_data_filtered.index.month])
# monthly_returns = monthly_grouped.apply(lambda group: (group.iloc[-1] - group.iloc[0]) / group.iloc[0])

# standar_plotting_monthly_yearly_heatmap_chart = standar_plotting_instance.get_monthly_yearly_heatmap_chart(
#     data=monthly_returns,
#     variable_list=monthly_returns.columns,
#     title=f"Monthly Return for the Ticker: {list(monthly_returns.columns)} over the time range: {start_date.strftime('%d-%m-%Y')} - {end_date.strftime('%d-%m-%Y')}",
#     xaxis_title=f"Year",
#     yaxis_title=f"Month"
#     )
# standar_plotting_monthly_yearly_heatmap_chart.show(renderer="browser")

# # Create a basic bar plot
# standard_plotting_bar_chart = standar_plotting_instance.get_bar_chart(
#     data=test_data_filtered.resample("YE").mean(),
#     variable_list=test_data_filtered.columns,
#     title=f"Yearly Mean of Adj Close for the Ticker: {list(test_data_filtered.columns)} over the time range: {start_date.strftime('%d-%m-%Y')} - {end_date.strftime('%d-%m-%Y')}",
#     xaxis_title="Year",
#     yaxis_title="Mean Adj Close"
#     )
# standard_plotting_bar_chart.show(renderer="browser")
# # Createa basic return histogram with an overlayed normal distribution
# standard_plotting_return_histogram = standar_plotting_instance.get_return_histogram(
#     data=test_data_filtered.pct_change().dropna(),
#     variable_list=test_data_filtered.columns,
#     title=f"Daily Return of Adj Close for the Ticker: {list(test_data_filtered.columns)} over the time range: {start_date.strftime('%d-%m-%Y')} - {end_date.strftime('%d-%m-%Y')}",
#     xaxis_title="Return value range",
#     yaxis_title="Absolute Frequency"
#     )
# standard_plotting_return_histogram.show(renderer="browser")
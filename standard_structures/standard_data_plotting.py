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
                self.__check_path_existence(path=graph_full_save_path)
                html_fig.write_html(graph_full_save_path)
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
                self.__check_path_existence(path=graph_full_save_path)
                html_fig.write_html(graph_full_save_path)
            if return_html_version:
                return fig, html_fig
            else:
                return fig
            
    
    def lighten_color(
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
            lightening_factor: float = 0.6
            ) -> List[str]:
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

            center_color = "#ffffff"  # divergence midpoint (white)

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
                self.__check_path_existence(path=graph_full_save_path)
                html_fig.write_html(graph_full_save_path)
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
                self.__check_path_existence(path=graph_full_save_path)
                html_fig.write_html(graph_full_save_path)
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
                self.__check_path_existence(path=graph_full_save_path)
                html_fig.write_html(graph_full_save_path)
            if return_html_version:
                return hist_fig, html_fig
            else:
                return hist_fig
    


# # Example usage
# PROJECT_PATH = r""
# RESULTS_PATH_ENDING = r"results/"
# GRAPHS_PATH_ENDING = r"graphs/"
# CAU_COLOR_SCALE = ["#9b0a7d", "grey", "black", "grey"]
# COLOR_DISCRETE_SEQUENCE_DEFAULT = CAU_COLOR_SCALE

# standar_plotting_instance = StandardPlotting(
#     project_path=PROJECT_PATH,
#     results_path_ending=RESULTS_PATH_ENDING,
#     graphs_path_ending=GRAPHS_PATH_ENDING,
#     color_discrete_sequence_default=COLOR_DISCRETE_SEQUENCE_DEFAULT
#     )
# # Create a basic plotly line chart, plotting all columns as separate lines
# standard_plotting_line_chart = standar_plotting_instance.get_line_chart(
#     data=test_data_filtered,
#     variable_list=test_data_filtered.columns,
#     title=f"Adj Close over the time range: {start_date.strftime('%d-%m-%Y')} - {end_date.strftime('%d-%m-%Y')}",
#     xaxis_title="Date",
#     yaxis_title="Adj Close",
#     legend_title="Stock Ticker"
# )
# standard_plotting_line_chart.show(renderer="browser")
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
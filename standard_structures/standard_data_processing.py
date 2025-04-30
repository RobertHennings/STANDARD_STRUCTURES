from typing import Dict, List
import pandas as pd
import io

class StandardDataProcessing(object):
    def __init__(
            self
            ):
        pass


    def convert_dataframe_to_csv(
            self,
            df: pd.DataFrame
            ):
            return df.to_csv(index=False).encode('utf-8')


    def convert_dataframe_to_excel(
            self,
            df: pd.DataFrame,
            sheet_name: str
            ):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            return buffer


    def get_max_drawdown(
            self,
            data: pd.DataFrame,
            window_size: int,
            return_simple_drawdown: bool
            ) -> pd.DataFrame:
            drawdown = data / data.rolling(window_size, min_periods=1).max() - 1.0
            max_drawdown = drawdown.rolling(30, min_periods=1).min()

            if return_simple_drawdown:
                return drawdown, max_drawdown

            print("FUNCTION: get_max_drawdown")
            return max_drawdown


    # Check after data has been loaded for some Yahoo Tickers
    # that there are no major series of NaNs that would be thrown out if: data.dropna()
    def check_data_columns_for_nan(
            self,
            data: pd.DataFrame
            ) -> pd.DataFrame:
            pass
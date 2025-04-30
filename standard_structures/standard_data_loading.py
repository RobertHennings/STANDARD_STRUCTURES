from typing import Dict, List
import os
import json
import pandas as pd
import datetime as dt
import yfinance as yf
import meteostat
from meteostat import Stations, Hourly
import financedatabase as fd
from energyquantified import EnergyQuantified

FUTURE_MONTH_MAPPING_DICT = {"January": "F",
                    "February": "G",
                    "March": "H",
                    "April": "J",
                    "May": "K",
                    "June": "M",
                    "July": "N",
                    "August": "Q",
                    "September": "U",
                    "October": "V",
                    "November": "X",
                    "December": "Z"}

class StandardDataLoading(object):
    def __init__(
            self,
            credential_path: str=None,
            credential_file_name: str=None,
            future_month_mapping_dict: Dict[str, str]=FUTURE_MONTH_MAPPING_DICT,
            ):
        self.credential_path = credential_path
        self.credential_file_name = credential_file_name
        if (self.credential_path and self.credential_file_name) is not None:
            # trigger credential loading
            credentials = self.__load_credentials()
            self.credentials = credentials
            # initialize the different API Clients
            eq_api_client = credentials.get("EnergyQuantified_API", None)
            entsoe_api_client = credentials.get("EntsoE_API", None)

            if eq_api_client is not None:
                self.eq_api_client = eq_api_client
            else:
                print(f"EnergyQuantified_API not defined in credentials file: {self.credential_file_name} in path: {self.credential_path}")
            if entsoe_api_client is not None:
                self.entsoe_api_client = entsoe_api_client
            else:
                print(f"EntsoE_API not defined in credentials file: {self.credential_file_name} in path: {self.credential_path}")


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

    def __load_credentials(
              self
              ):
        # First check if a secrets file is already present at the provided path
        if self.credential_file_name is not None and self.credential_path is not None:
            self.__check_path_existence(path=self.credential_path)
            if self.credential_file_name in os.listdir(self.credential_path):
                file_path_full = f"{self.credential_path}/{self.credential_file_name}"
                with open(file_path_full, encoding="utf-8") as json_file:
                    credentials = json.load(json_file)
                return credentials
            else:
                raise KeyError(f"{self.credential_file_name} not found in path: {self.credential_path}")
        else:
            print("No credentials provided, missing the file path: {} and/or the file name: {}")


    def get_yahoo_data(
        self,
        ticker: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        auto_adjust: bool=False
        ) -> pd.DataFrame:
        tickers_connected = " ".join(ticker)
        print("FUNCTION: get_yahoo_data")
        data = yf.download(tickers_connected, start=start_date, end=end_date, auto_adjust=auto_adjust)
        return data


    def filter_yahoo_data(
        self,
        yahoo_df: pd.DataFrame,
        columns: List[str]
        ) -> pd.DataFrame:
        print("FUNCTION: filter_yahoo_data")
        return yahoo_df[columns]


    def get_news_df(
            self,
            ticker: str
            ) -> pd.DataFrame:
        try:
            news_list_dict = yf.Ticker(ticker).get_news()
            news_list_dict = [news.get("content") for news in news_list_dict]
            news_df = pd.DataFrame(news_list_dict)
            news_df = news_df[['title', 'summary', 'pubDate', 'provider']]
            news_df['provider_name'] = news_df['provider'].apply(lambda x: x.get('displayName'))
            news_df['provider_url'] = news_df['provider'].apply(lambda x: x.get('url'))

            # Drop the original nested column if needed
            news_df = news_df.drop(columns=['provider'])
            news_df.pubDate = pd.to_datetime(news_df.pubDate)
            news_df = news_df.set_index("pubDate", drop=True).sort_index(ascending=False)

            # Renaming columns for showing
            news_df.columns = [col.title() for col in news_df.columns]
            news_df.index.name = "Published"
        except:
            news_df = pd.DataFrame()
        print("FUNCTION: get_news_df")
        return news_df


    def get_actual_indices_symbol(
            self,
            return_name: bool=False,
            ) -> list:
        # CAUTION: Watch the specific installed version, might downgrade
        try:
            indices = fd.Indices()
            act_indices_symbol = indices.data.index.to_list()
            act_indices_name = indices.data.name.to_list()
            return act_indices_symbol if not return_name else act_indices_name
        except:
            print("Actualisation from Fin Database currently not possible")


    def get_meteostat_data_hourly(
            self,
            lat: float,
            lon: float,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp,
            variable_list: List[str]=None
            ) -> pd.DataFrame:
        # load all available stations
        stations = Stations()
        # determine the stations nearest to the given point
        stations = stations.nearby(lat, lon)
        # select the nearest station
        num_stations = 1
        station = stations.fetch(num_stations)
        station_info = station[["name", "country", "region", "latitude", "longitude", "elevation"]]
        weather_data = Hourly(station, start_date, end_date)
        weather_data = weather_data.fetch()

        if variable_list is not None:
            weather_data = weather_data[variable_list]
        print(f"Station data:\n{station_info.T}")
        return weather_data


    def get_eq_data(
            self,
            curve: str,
            country: str=None,
            start_date: pd.Timestamp=dt.datetime.today() - pd.Timedelta(days=30),
            end_date: pd.Timestamp=dt.datetime.today()
            ) -> pd.Series:
        if self.eq_api_client is not None:
            eq_api_client = self.eq_api_client
        else:
            raise Exception("EnergyQuantified_API API Key not properly provided")
        if self.eq_api_client is not None:
            # build up the curve name
            if country is not None:
                curve_name = f"{country.upper()} {curve.capitalize()}"
            else:
                curve_name = curve
            print(f"Loading: {curve_name}")
            # Check what type of curve has to be loaded
            if str(eq_api_client.metadata.curve(curve_name).curve_type) == "INSTANCE":
                eq_data = eq_api_client.instances.latest(curve_name).to_df()
            else:
                eq_data = eq_api_client.timeseries.load(
                    curve_name,
                    begin=start_date,
                    end=end_date).to_df()
            eq_data.columns = [curve]
            return eq_data


    def get_futures_ticker(
            self,
            product_abbrev: str,
            years: int
            ) -> list:
        fut_month_dict = self.future_month_mapping_dict
        # Determine current date
        actual_month = fut_month_dict[(dt.datetime.today() + pd.offsets.MonthBegin(1)).month_name()]
        # Get index of actual month
        actual_month_index = list(fut_month_dict.values()).index(actual_month)
        crude_oil_futures = []
        # Create full list
        for year in pd.date_range(dt.datetime.today(), periods=years, freq="Y").strftime("%y"):
            for abbrev in fut_month_dict.values():
                crude_oil_futures.append(f"{product_abbrev}{abbrev}{year}.NYM")
                # print(f"CL{abbrev}{year}.NYM")
        # Drop the old contracts
        crude_oil_futures = crude_oil_futures[actual_month_index:]
        return crude_oil_futures


# Example usage
# standard_data_loading_instance = StandardDataLoading(
#     credential_path="credentials.json",
#     credential_file_name=r"/Users/Robert_Hennings/Dokumente/IT Weiterbildung/Python Data Analysis /AbbeV/Meine Projekte"
#     )
# ticker = ["^GDAXI", "^DJI", "^N225", "^GSPC", "^NYA", "^N100"]
# start_date = (dt.datetime.today() - pd.Timedelta(days=365*5))
# end_date = dt.datetime.today()
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

# # Amount of NaN values diretly in %
# (test_data_filtered.isna().sum(axis=0) / test_data_filtered.shape[0]) * 100

# standard_loading_indices_names = standard_data_loading_instance.get_actual_indices_symbol(
#     return_name=True
#     )
# standard_loading_indices_symbols = standard_data_loading_instance.get_actual_indices_symbol(
#     return_name=False
#     )
# standard_loading_meteostat_data_hourly = standard_data_loading_instance.get_meteostat_data_hourly(
#     lat=54.40949,
#     lon=10.22698,
#     start_date=start_date,
#     end_date=end_date
#     )

# etfs = fd.ETFs()
# fonds_name = "ARERO - Der Weltfonds"
# fonds_name in etfs.data.name.to_list()
# etfs.data.query("name == @fonds_name")
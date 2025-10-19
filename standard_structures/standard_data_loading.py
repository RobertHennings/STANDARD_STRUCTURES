from typing import Dict, List
import os
import zipfile
import io
import json
import pandas as pd
import datetime as dt
import yfinance as yf
import meteostat
from meteostat import Stations, Hourly
import financedatabase as fd
from energyquantified import EnergyQuantified
import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
REQUEST_TIMEOUT = 90
ERROR_CODES_DICT = {
        400: {
            "message": "Bad request",
            "return_": None
        },
        401: {
            "message": "Failed to authenticate, check your authenticat…",
            "return_": None
        },
        410: {
            "message": "Unauthorized",
            "return_": None
        },
        403: {
            "message": "Forbidden",
            "return_": None
        },
        404: {
            "message": "Not Found",
            "return_": None
        },
        405: {
            "message": "Method not allowed",
            "return_": None
        },
        406: {
            "message": "Not acceptable",
            "return_": None
        },
        409: {
            "message": "Conflict",
            "return_": None
        },
        415: {
            "message": "Unsopprted Media Type",
            "return_": None
        },
        500: {
            "message": "Internal Server Error",
            "return_": None
        },
        502: {
            "message": "Bad Gateway",
            "return_": None
        },
        503: {
            "message": "Service Unavailable",
            "return_": None
        },
        504: {
            "message": "Gateway Timeout",
            "return_": None
        }
    }
SUCCESS_CODES_DICT = {
    200: {
        "message": "OK",
        "return_": "Success"
        },
    201: {
    "message": "Created",
    "return_": "Success"
    },
    204: {
        "message": "No content",
        "return_": "Success"
        }
}

class StandardDataLoading(object):
    def __init__(
            self,
            credential_path: str=None,
            credential_file_name: str=None,
            proxies: Dict[str, str]=None,
            verify: bool=None,
            future_month_mapping_dict: Dict[str, str]=FUTURE_MONTH_MAPPING_DICT,
            request_timeout: int=REQUEST_TIMEOUT,
            error_codes_dict: Dict[int, Dict[str, str]]=ERROR_CODES_DICT,
            success_codes_dict: Dict[int, Dict[str, str]]=SUCCESS_CODES_DICT,
            ):
        self.credential_path = credential_path
        self.credential_file_name = credential_file_name
        self.future_month_mapping_dict = future_month_mapping_dict
        self.request_timeout = request_timeout
        self.error_codes_dict = error_codes_dict
        self.success_codes_dict = success_codes_dict
        self.proxies = proxies
        self.verify = verify
        if (self.credential_path and self.credential_file_name) is not None:
            # trigger credential loading
            credentials = self.__load_credentials()
            self.credentials = credentials
            # initialize the different API Clients
            eq_api_client = credentials.get("EnergyQuantifiedAPI", None)
            entsoe_api_client = credentials.get("EntsoeAPI", None)

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
            logging.info(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            logging.info(f"Folder: {folder_name} created in path: {path}")


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
            logging.info(f"No credentials provided, missing the file path: {self.credential_path} and/or the file name: {self.credential_file_name}")


    def __connection_handler(
        self,
        response: requests.models.Response
        ):
        """Testing the response and checking for robustness of returned
           content.

        Args:
            response (requests.models.Response): direct requests response

        Raises:
            Exception: Error Message
        """
        # Test the established connection for robustness and validity
        connector_response = response
        status_code = connector_response.status_code
        response_url = connector_response.url
        response_headers = connector_response.headers
        # Define the potential occuring errors
        errors_msgs = {
            304: "No changes. There have been no changes to the data since the timestamp supplied in the If-Modified-Since header.",
            400: "Syntax error. Syntactic or semantic issue with the parameters supplied.",
            404: "No results found. There are no results matching the query.",
            406: "Not Acceptable.",
            500: "Internal Server Error. Feel free to try again later or to contact the support hotline https://ecb-registration.escb.eu/statistical-information.",
            501: "Not implemented.",
            503: "Service unavailable: Web service is temporarily unavailable.",
            }
        if status_code == 200:
            logging.info(f'Connection established with url: {response_url} and headers: {response_headers}')
        else:
            if status_code in errors_msgs:
                raise Exception(f'Request Error Code: {status_code} - {errors_msgs.get(status_code, "")}')
            else:
                logging.error(f'Unexpected Error: {response_url} - {response_headers}')
                logging.error(f'Request Error Code: {status_code}')
                connector_response.raise_for_status()


    ### Further robustness checks
    def __resilient_request(
            self,
            response: requests.models.Response,
            ):
        """Internal helper method - serves as generous requests
           checker using the custom defined error and sucess code dicts
           for general runtime robustness

        Args:
            response (requests.models.Response): generous API response

        Raises:
            Exception: _description_

        """
        status_code = response.status_code
        response_url = response.url
        status_code_message = [
            dict_.get(status_code).get("message")
            for dict_ in [self.error_codes_dict, self.success_codes_dict]
            if status_code in dict_.keys()]
        # If status code is not present in the defined dicts
        if status_code_message == []:
            logging.info(f"Status code: {status_code} not defined")
        else: # if status code is definded in the dicts
            # get the defined message for the status code
            status_code_message = f"{"".join(status_code_message)} for URL: {response_url}"
            # get the defined return (type) for the status code
            status_code_return = [
                dict_.get(status_code).get("return_")
                for dict_ in [self.error_codes_dict, self.success_codes_dict]
                if status_code in dict_.keys()]

            if status_code_return is not None:
                logging.info(status_code_message)
            else:
                raise Exception("Error")

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
        # Drop the old contracts
        crude_oil_futures = crude_oil_futures[actual_month_index:]
        return crude_oil_futures


    def get_futures_curve(
        self,
        futures_ticker: List[str],
        future_type: str,
        fut_month_dict: Dict[str, str]=FUTURE_MONTH_MAPPING_DICT
        ) -> pd.DataFrame:

        fut_month_dict_rev = {new_key: new_value for new_key, new_value in zip(fut_month_dict.values(), fut_month_dict.keys())}
        # build yahoo finance query for data loading
        crude_oil_futures_str = " ".join(futures_ticker)
        ticker_data = yf.download(
            tickers=crude_oil_futures_str,
            period="1d",
            group_by="ticker",
            auto_adjust=False
            )
        ticker_data_master = pd.DataFrame(
            index=ticker_data.columns.get_level_values(level=0).unique()
            ).reset_index(drop=False)

        for level in ticker_data.columns.get_level_values(level=1).unique():
            ticker_data_master[level] = ticker_data.groupby(level=1, axis=1).get_group(level).T.values

        months = ticker_data_master.Ticker.str.split(".").str[0].str[-3:].apply(lambda x: fut_month_dict_rev.get(x[0]))
        years = ticker_data_master.Ticker.str.split(".").str[0].str[-3:].str[1:]

        contract_expiry = [pd.Timestamp(f"01 {month} {year}") for month, year in zip(months, years)]
        ticker_data_master["Expiry"] = contract_expiry
        ticker_data_master = ticker_data_master.sort_values("Expiry")
        ticker_data_master["Date"] = dt.datetime.today().strftime("%Y-%m-%d")
        ticker_data_master["Underlying"] = future_type
        return ticker_data_master


    def get_cftc_commitment_of_traders(
        self,
        start_date: str,
        end_date: str,
        base_url: str=r"https://www.cftc.gov/files/dea/history/",
        report_type: str="Futures-and-Options Combined Reports",
        save_files_locally: bool=False,
        save_path: str=None,
        file_ending_dict: Dict[str, str]=None,
        ) -> pd.DataFrame:
        """
        Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm#:~:text=(Text%2C%20Excel)-,Futures%2Dand%2DOptions%20Combined%20Reports:,the%20new%20format%20by%20year.

        Args:
            base_url (str): _description_
            start_date (str): _description_
            end_date (str): _description_
            report_type (str, optional): _description_. Defaults to "Futures-and-Options Combined Reports".
            save_files_locally (bool, optional): _description_. Defaults to False.
            save_path (str, optional): _description_. Defaults to None.
            file_ending_dict (Dict[str, str], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_

        Examples:

        """
        year_list_files = pd.date_range(start=start_date, end=end_date, freq="YS").year.to_list()
        file_ending_dict = {
            "Futures-and-Options Combined Reports": "deahistfo",
            "Commodity Index Trader Supplement": "",
            "Traders in Financial Futures ; Futures-and-Options Combined Reports": "",
            "Traders in Financial Futures ; Futures Only Reports": "",
            "Disaggregated Futures-and-Options Combined Reports": "",
            "Disaggregated Futures Only Reports": ""
        }
        file_ending = file_ending_dict.get(report_type, None)
        if file_ending is not None:
            url_list = [f"{base_url}{file_ending}_{year}.zip" for year in year_list_files]
        else:
            raise ValueError(f"Report type '{report_type}' not supported.")
        combined_data = []
        # https://www.cftc.gov/files/dea/history/deahistfo_2004.zip
        # url = "https://www.cftc.gov/files/dea/history/deahistfo_2005.zip"
        # url = url_list[0]
        # Retrieve the zip files from the urls and save them locally
        for url in url_list:
            # logging.info(f"Downloading file from URL: {url}")
            print(f"Downloading file from URL: {url}")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                try:
                    # logging.info(f"File for year {url.split('_')[-1].split('.')[0]} failed, trying different naming convention.")
                    print(f"File for year {url.split('_')[-1].split('.')[0]} failed, trying different naming convention.")
                    url_changed = "".join(url.split('_'))
                    print(f"Trying URL: {url_changed}")
                    response = requests.get(url_changed, timeout=REQUEST_TIMEOUT)
                    if response.status_code == 200:
                        print(f"File for year {url.split('_')[-1].split('.')[0]} successfully loaded with changed URL")
                except:
                    print(f"File for year not found")
            # self.__connection_handler(response=response)
            # self.__resilient_request(response=response)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_name in z.namelist():
                    # logging.info(f"Extracting file: {file_name}")
                    print(f"Extracting file: {file_name}")
                    with z.open(file_name) as file:
                        # Assuming the file is a CSV, adjust if it's another format
                        df = pd.read_csv(file)
                        combined_data.append(df)

        # Combine all data into a single DataFrame
        if combined_data != []:
            final_df = pd.concat(combined_data, ignore_index=True).reset_index(drop=True)
            final_df["Market and Exchange Names"] = final_df["Market and Exchange Names"].str.strip()
        else:
            logging.warning("No data was extracted from the zip files.")
            final_df = pd.DataFrame()
        if save_files_locally and save_path is not None:
            self.__check_path_existence(path=save_path)
            file_name = f"{report_type}_{start_date}_to_{end_date}.csv"
            file_path_full = f"{save_path}/{file_name}"
            final_df.to_csv(file_path_full, index=False)
            logging.info(f"Data saved locally at: {file_path_full}")
        return final_df


# Example usage
# standard_data_loading_instance = StandardDataLoading(
#     credential_path="credentials.json",
#     credential_file_name=None
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
# Loading CFTC Commitment of Traders Report data
# start_date = "1995-01-01"
# end_date = "2025-01-01"
# report_type = "Futures-and-Options Combined Reports"
# save_files_locally = False
# save_path = r"/Users/Robert_Hennings/Uni/Master/Seminar/reports/figures"

# cftc_data = standard_data_loading_instance.get_cftc_commitment_of_traders(
#     start_date=start_date,
#     end_date=end_date,
#     report_type=report_type,
#     save_files_locally=save_files_locally,
#     save_path=save_path
# )
# cftc_data["As of Date in Form YYYY-MM-DD"] = pd.to_datetime(cftc_data["As of Date in Form YYYY-MM-DD"])

# oil_gas_products_list = [
#                          "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
#                          "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
#                          "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE",
#                          "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE"
#                          ]

# cftc_data_oil_gas = cftc_data[
#     cftc_data["Market and Exchange Names"].isin(oil_gas_products_list)
# ].sort_values(by=["As of Date in Form YYYY-MM-DD"]).reset_index(drop=True)

# etfs = fd.ETFs()
# fonds_name = "ARERO - Der Weltfonds"
# fonds_name in etfs.data.name.to_list()
# etfs.data.query("name == @fonds_name")
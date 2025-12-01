import json
import os
 
from newsplease import NewsPlease
import pandas as pd
import pickle
import requests
import tldextract
from tqdm import tqdm
import zipfile
import numpy as np
 
 # This file is used to get data from GDELT and label it
 
 
if __name__=="__main__":
 
    years = ["2023"]#
    months = ["10"]#,"02","03","04","05","06","07","08","09","10","11","12"]
    days = [["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"],
            # ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
            ]
    hours = ["00","01","02","03","04","05","06","07","08","09","10","11","12"]
    minutes = ["0000","1500","3000","4500"]
    
    for year in years:
        for i in range(len(months)):
            month = months[i]
            days_of_month = days[i]
            for day in tqdm(days_of_month):
                for hour in hours:
                    for minute in minutes:
                        try: 
                            url_export = "http://data.gdeltproject.org/gdeltv2/"+year+month+day+hour+minute+".export.CSV.zip"
                            url_mentions = "http://data.gdeltproject.org/gdeltv2/"+year+month+day+hour+minute+".mentions.CSV.zip"

                            r_export = requests.get(url_export, allow_redirects=True)
                            r_mentions = requests.get(url_mentions, allow_redirects=True)
                            
                            r_export.raise_for_status()  
                            r_mentions.raise_for_status()
                            
                            open('./'+year+month+day+hour+minute+".export.CSV.zip", 'wb').write(r_export.content)
                            open('./'+year+month+day+hour+minute+".mentions.CSV.zip", 'wb').write(r_mentions.content)

                            with zipfile.ZipFile('./'+year+month+day+hour+minute+".export.CSV.zip", 'r') as zip_ref:
                                zip_ref.extractall("./gdelt_data_event_export/")
                            with zipfile.ZipFile('./'+year+month+day+hour+minute+".mentions.CSV.zip", 'r') as zip_ref:
                                zip_ref.extractall("./gdelt_data_event_mentions/")
                           
                            os.remove('./'+year+month+day+hour+minute+".export.CSV.zip")
                            os.remove('./'+year+month+day+hour+minute+".mentions.CSV.zip")
                        except Exception as e:
                            print(f"Failed for {year+month+day+hour+minute}: {e}")
 
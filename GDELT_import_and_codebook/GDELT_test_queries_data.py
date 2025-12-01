import pandas as pd
from pandasql import sqldf
import glob
import os

# Define the directories containing the data
PATH_MENTIONS = "./gdelt_data_event_mentions/"
PATH_EXPORT = "./gdelt_data_event_export/"

# Column definitions
COL_NAMES_MENTIONS = [
    "GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType",
    "MentionSourceName", "MentionIdentifier", "SentenceID",
    "Actor1CharOffset", "Actor2CharOffset", "ActionCharOffset",
    "InRawText", "Confidence", "MentionDocLen", "MentionDocTone",
    "SRCLC", "ENG"
]

COL_NAMES_EVENTS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_Fullname", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_Fullname", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat",
    "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_Fullname", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat",
    "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
]

# --- Function to load and concatenate all files from a directory ---
def load_gdelt_files(directory, file_pattern, col_names, dtype_dict):
    # Construct the full search path
    search_path = os.path.join(directory, file_pattern)
    files = glob.glob(search_path)[0:2]
    
    if not files:
        print(f"Warning: No files found in {search_path}")
        return pd.DataFrame(columns=col_names)
    
    print(f"Found {len(files)} files in {directory}. Loading...")
    
    df_list = []
    for filename in files:
        try:
            # Check if filename roughly matches the timestamp pattern (simple digit check)
            # The pattern is usually YYYYMMDDHHMMSS at the start of the filename
            basename = os.path.basename(filename)
            if basename[0].isdigit(): 
                df = pd.read_csv(
                    filename,
                    sep="\t",
                    names=col_names,
                    dtype=dtype_dict,
                    on_bad_lines='skip' # Optional: skip bad lines to prevent crashes
                )
                df_list.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame(columns=col_names)

# --- Load Events Data ---
print("--- Loading Events ---")
events = load_gdelt_files(
    PATH_EXPORT, 
    "*.export.CSV", 
    COL_NAMES_EVENTS, 
    {"GlobalEventID": "string", "EventCode": "string"}
)
print(f"Total Events loaded: {len(events)}")

# --- Load Mentions Data ---
print("--- Loading Mentions ---")
mentions = load_gdelt_files(
    PATH_MENTIONS, 
    "*.mentions.CSV", 
    COL_NAMES_MENTIONS, 
    {"GlobalEventID": "string"}
)
print(f"Total Mentions loaded: {len(mentions)}")

# Define environment for SQL
env = {
    "events": events,
    "mentions": mentions,
}

pysqldf = lambda q: sqldf(q, env)

# Load the query dataset
dataset_path = "./GDELT_import_and_codebook/new_sql_queries_for_testing.csv"
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found.")
    dataset = pd.DataFrame()
else:
    dataset = pd.read_csv(dataset_path, encoding="utf-8")

results_summary = []

# Process queries
if not dataset.empty:
    for idx, row in dataset.iterrows():
        query_id = row.get("id", idx)
        text_query = row.get("text_query", "")
        sql_query = row["sql_query"]

        print(f"\n=== Test de la requête id={query_id} ===")
        print(f"Texte : {text_query}")

        try:
            df_res = pysqldf(sql_query)

            status = "success"
            error_msg = ""
            n_rows = len(df_res)
            n_cols = len(df_res.columns)

            print(f"--> OK, {n_rows} lignes, {n_cols} colonnes")
            if n_rows > 0:
                print(df_res.head(3))

        except Exception as e:
            status = "error"
            error_msg = str(e)
            n_rows = None
            n_cols = None

            print(f"--> ERREUR pour la requête id={query_id}: {error_msg}")

        results_summary.append(
            {
                "id": query_id,
                "text_query": text_query,
                "sql_query": sql_query,
                "status": status,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "error": error_msg,
            }
        )

    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("./GDELT_import_and_codebook/new_sql_results_summary_pandasql.csv", index=False, encoding="utf-8")

    print("\n=== Terminé ===")
    print("Résumé des résultats sauvegardé dans new_results_summary_pandasql.csv")
else:
    print("Dataset empty or not found. Exiting.")

errors_df = results_df[results_df['status'] == 'error']
    
if not errors_df.empty:
    error_filename = "./GDELT_import_and_codebook/new_queries_error.csv"
    errors_df.to_csv(error_filename, index=False, encoding="utf-8")
    print(f"⚠️ {len(errors_df)} failed queries saved to {error_filename}")
else:
    print("✅ No errors found. No error file created.")

print("\n=== Terminé ===")
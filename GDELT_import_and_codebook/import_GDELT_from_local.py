import pandas as pd
from pandasql import sqldf

# import GDELT data with the right column after requesting it from the website, and testing the generated SQL queries with pandasql

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

events = pd.read_csv(
    "20211102144500.export.CSV",
    sep="\t",
    names=COL_NAMES_EVENTS,
    dtype={"GlobalEventID": "string", "EventCode": "string"},
)

mentions = pd.read_csv(
    "20211102144500.mentions.CSV",
    sep="\t",
    names=COL_NAMES_MENTIONS,
    dtype={"GlobalEventID": "string"},
)

env = {
    "events": events,
    "mentions": mentions,
}

pysqldf = lambda q: sqldf(q, env)



dataset = pd.read_csv("ttsql_training_dataset_GDELT.csv", encoding="utf-8")

results_summary = []

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
results_df.to_csv("ttsql_results_summary_pandasql.csv", index=False, encoding="utf-8")

print("\n=== Terminé ===")
print("Résumé des résultats sauvegardé dans ttsql_results_summary_pandasql.csv")

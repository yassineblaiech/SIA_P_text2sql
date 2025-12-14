import os
import glob
import pandas as pd
from pandasql import sqldf
from io import StringIO
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# PART 1: THE VALIDATOR ENGINE 
# ==========================================

class GDELTValidator:
    def __init__(self, path_mentions, path_export):
        self.path_mentions = path_mentions
        self.path_export = path_export
        self.events = None
        self.mentions = None
        self.env = {}
        
        # Column Definitions (From your script)
        self.COL_NAMES_MENTIONS = [
            "GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType",
            "MentionSourceName", "MentionIdentifier", "SentenceID",
            "Actor1CharOffset", "Actor2CharOffset", "ActionCharOffset",
            "InRawText", "Confidence", "MentionDocLen", "MentionDocTone",
            "SRCLC", "ENG"
        ]

        self.COL_NAMES_EVENTS = [
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

    def load_data(self):
        """Loads the CSVs into memory once."""
        print("--- Loading Events ---")
        self.events = self._load_files(
            self.path_export, 
            "*.export.CSV", 
            self.COL_NAMES_EVENTS, 
            {"GlobalEventID": "string", "EventCode": "string"}
        )
        print(f"Total Events loaded: {len(self.events)}")

        print("--- Loading Mentions ---")
        self.mentions = self._load_files(
            self.path_mentions, 
            "*.mentions.CSV", 
            self.COL_NAMES_MENTIONS, 
            {"GlobalEventID": "string"}
        )
        print(f"Total Mentions loaded: {len(self.mentions)}")
        
        # Prepare SQL environment
        self.env = {"events": self.events, "mentions": self.mentions}

    def _load_files(self, directory, file_pattern, col_names, dtype_dict):
        search_path = os.path.join(directory, file_pattern)
        files = glob.glob(search_path)[0:2] # Keeping your limit of 2 files for speed
        
        if not files:
            print(f"Warning: No files found in {search_path}")
            return pd.DataFrame(columns=col_names)
        
        df_list = []
        for filename in files:
            try:
                basename = os.path.basename(filename)
                if basename[0].isdigit():
                    df = pd.read_csv(
                        filename, sep="\t", names=col_names, dtype=dtype_dict, on_bad_lines='skip'
                    )
                    df_list.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if df_list:
            return pd.concat(df_list, ignore_index=True)
        return pd.DataFrame(columns=col_names)

    def validate_query(self, sql_query):
        """Runs a single query and returns (Success, ErrorMessage)"""
        try:
            # We use pandasql to execute the query against the loaded dataframes
            sqldf(sql_query, self.env)
            return True, ""
        except Exception as e:
            return False, str(e)

# ==========================================
# PART 2: LANGCHAIN SETUP
# ==========================================

# Initialize LLM
llm = ChatOpenAI(
    model="deepseek-reasoner",              
    temperature=0.7,
    base_url="https://api.deepseek.com/v3.2_speciale_expires_on_20251215",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# 1. GENERATION PROMPT
gen_system_prompt = """
You are a SQL expert specializing in the GDELT 2.0 Dataset.
Your goal is to generate unique, analytical SQL queries based on the provided GDELT Codebook.
Output must be a valid CSV format with columns: id, text_query, sql_query.
Use a semicolon (,) as the CSV separator to avoid conflicts with SQL commas.
Quote all fields with double quotes.
you should write text_query in French. Also, make the text_query as simple and clear as possible (don't use the name of columns in the text_query).

Dataset Rules:
1. Tables are named 'events' and 'mentions'.
2. Join them on 'GlobalEventID'.
3. Use standard SQLite syntax (supported by pandasql).
4. Do NOT use functions like STDDEV (use simple math or approximations).

GDELT Schema:
{codebook}

Example Valid Queries:
{examples}

So make sure to seperate columns with , and not ;.
Also, make sure to quote all fields with double quotes.
"""

gen_user_prompt = "Generate {n} distinct, complex SQL queries (starting ID: {start_id})."

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", gen_system_prompt),
    ("user", gen_user_prompt)
])

# 2. CORRECTION PROMPT
fix_system_prompt = """
You are a SQL Debugging Assistant. 
You will receive a CSV row containing a SQL query that FAILED to execute on a SQLite engine.
You will also receive the specific Error Message.
Your job is to rewrite the SQL query to fix the error while maintaining the original intent.

Common Errors to fix:
- "no such column": Check if the column exists in the GDELT 2.0 schema.
- "no such function": SQLite has limited functions (no STDDEV, no CONCAT in some versions, use ||).
- Syntax errors.

Output ONLY the corrected CSV row ("id","text_query","sql_query") using , separators and not ;.
"""

fix_user_prompt = """
Broken Query Row: {row}
Error Message: {error}
"""

fix_prompt = ChatPromptTemplate.from_messages([
    ("system", fix_system_prompt),
    ("user", fix_user_prompt)
])

# Chains
gen_chain = gen_prompt | llm | StrOutputParser()
fix_chain = fix_prompt | llm | StrOutputParser()

# ==========================================
# PART 3: MAIN ORCHESTRATION
# ==========================================

def run_pipeline():
    # 1. SETUP DATA
    # Adjust paths to your actual folders
    PATH_MENTIONS = "./gdelt_data_event_mentions/" 
    PATH_EXPORT = "./gdelt_data_event_export/"
    
    validator = GDELTValidator(PATH_MENTIONS, PATH_EXPORT)
    validator.load_data()

    # 2. CONTEXT 
    # (You would paste the full codebook text here, truncated for brevity)
    try:
        with open("./GDELT_import_and_codebook/gdelt_schema.txt", "r", encoding="utf-8") as f:
            gdelt_codebook_text = f.read()
    except FileNotFoundError:
        print("Error: gdelt_codebook.txt not found. Using empty context.")
        gdelt_codebook_text = ""
    
    example_queries_text = """
1,"Liste les 10 événements qui ont le plus de mentions dans la presse, avec leur nombre de mentions et les acteurs principaux.","SELECT GlobalEventID, Actor1Name, Actor2Name, NumMentions FROM events ORDER BY NumMentions DESC LIMIT 10;"
2,"Donne le nombre d’événements par catégorie QuadClass (coopération verbale, coopération matérielle, conflit verbal, conflit matériel).","SELECT QuadClass, COUNT(*) AS nb_events FROM events GROUP BY QuadClass ORDER BY QuadClass;"
3,"Calcule le score moyen de Goldstein par pays de l’acteur 1, et retourne les 15 pays les plus “intenses” (en valeur absolue).","SELECT Actor1CountryCode, AVG(GoldsteinScale) AS avg_goldstein, COUNT(*) AS nb_events FROM events WHERE Actor1CountryCode IS NOT NULL GROUP BY Actor1CountryCode HAVING COUNT(*) >= 5 ORDER BY ABS(AVG(GoldsteinScale)) DESC LIMIT 15;"
4,"Donne les 10 événements ayant le ton moyen le plus négatif, avec les acteurs et le pays géographique principal.","SELECT GlobalEventID, Actor1Name, Actor2Name, Actor1Geo_Fullname, AvgTone FROM events WHERE AvgTone IS NOT NULL ORDER BY AvgTone ASC LIMIT 10;"
    """

    # 3. GENERATION STEP
    N_QUERIES = 200
    print(f"\n--- Generating {N_QUERIES} Queries ---")
    
    csv_output = gen_chain.invoke({
        "codebook": gdelt_codebook_text,
        "examples": example_queries_text,
        "n": N_QUERIES,
        "start_id": 1
    })
    print("Generated Queries CSV:\n", csv_output)

    # Parse CSV output into a list of dicts
    # Note: We expect LLM to output CSV with ; separator

    df_queries = pd.read_csv(StringIO(csv_output), sep=",", header=None, names=["id", "text_query", "sql_query"])
        

    print(f"Generated {len(df_queries)} queries.")

    # 4. VALIDATION & CORRECTION LOOP
    final_rows = []
    
    print("\n--- Validating Queries ---")
    
    for index, row in df_queries.iterrows():
        query_id = row['id']
        sql = row['sql_query']
        text = row['text_query']
        
        print(f"Checking ID {query_id}...", end=" ")
        
        is_valid, error_msg = validator.validate_query(sql)
        
        if is_valid:
            print("✅ Valid")
            final_rows.append(row)
        else:
            print(f"❌ Error: {error_msg}")
            print("   Attempting AI Repair...", end=" ")
            
            # --- CORRECTION STEP ---
            try:
                # Ask LLM to fix it
                csv_line = f'"{query_id}",{text},{sql}'
                fixed_csv_row = fix_chain.invoke({
                    "row": csv_line,
                    "error": error_msg
                })
                
                # Parse the fixed row
                # We assume LLM returns a single CSV line. We wrap it in StringIO to parse.
                # Adding header=None because it's just one row
                fixed_df = pd.read_csv(StringIO(fixed_csv_row), sep=",", header=None, names=["id", "text_query", "sql_query"])
                
                if not fixed_df.empty:
                    fixed_sql = fixed_df.iloc[0]['sql_query']
                    
                    # Double Check the fix
                    is_valid_now, error_msg_2 = validator.validate_query(fixed_sql)
                    
                    if is_valid_now:
                        print("✅ Fixed!")
                        final_rows.append(fixed_df.iloc[0])
                    else:
                        print(f"❌ Fix Failed ({error_msg_2}) - Discarding.")
            except Exception as e:
                print(f"❌ Repair Process Failed: {e}")

    # 5. EXPORT FINAL DATASET
    print("\n--- Exporting Final Dataset ---")

    new_df = pd.DataFrame(final_rows)
    output_path = "./GDELT_import_and_codebook/validated_gdelt_queries.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Found existing file at {output_path}. Appending new data...")
        try:
            existing_df = pd.read_csv(output_path)
            
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            print("Existing file was empty. Starting fresh.")
            final_df = new_df
    else:
        print("No existing file found. Creating new dataset...")
        final_df = new_df

    final_df.to_csv(output_path, index=False, sep=",", quoting=1) # quoting=1 ensures quote_all
    print(f"Saved {len(final_df)} total queries (Old + New) to {output_path}")

if __name__ == "__main__":
    run_pipeline()
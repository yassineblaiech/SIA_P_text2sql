import os
import glob
import pandas as pd
from pandasql import sqldf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib
import traceback

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

# ==========================================
# CONFIGURATION
# ==========================================

llm = ChatOpenAI(
    model="deepseek-chat",              
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0
)

INPUT_FILE = "./GDELT_import_and_codebook/validated_gdelt_queries.csv"
PATH_MENTIONS = "./gdelt_data_event_mentions/" 
PATH_EXPORT = "./gdelt_data_event_export/"
INPUT_QUERIES_FILE = "./GDELT_import_and_codebook/validated_gdelt_queries.csv"
OUTPUT_DATASET = "./GDELT_import_and_codebook/gdelt_viz_dataset.csv"

# ==========================================
# PART 1: DATA LOADER (FULL DATASET)
# ==========================================

def reindex_ids():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"--- Reading {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE, quotechar='"', skipinitialspace=True)
    
    print(f"Loaded {len(df)} rows.")
    print(f"Old IDs sample: {df['id'].head(5).tolist()}")

    df['id'] = range(1, len(df) + 1)
    
    print(f"New IDs sample: {df['id'].head(5).tolist()}")

    df.to_csv(INPUT_FILE, index=False, quoting=1)
    print(f"✅ Successfully re-indexed and overwrote {INPUT_FILE}")

    if os.path.exists(OUTPUT_DATASET):
        print("\n" + "="*50)
        print("⚠️  ACTION REQUIRED  ⚠️")
        print("You must DELETE or RENAME the file:")
        print(f"   {OUTPUT_DATASET}")
        print("Because the Input IDs have changed, the old 'resume' file is invalid.")
        print("="*50)

def test_readability():
    print(f"--- Testing Readability of {OUTPUT_DATASET} ---")
    
    try:
        df = pd.read_csv(OUTPUT_DATASET)
        
        print(f"✅ Successfully loaded CSV.")
        print(f"Total Rows: {len(df)}")
        print(f"Columns found: {list(df.columns)}")
        print("-" * 30)

        if df.empty:
            print("File is empty (contains only headers).")
            return

        first_row = df.iloc[0]
        
        print(f"Sample ID: {first_row['id']}")
        print(f"Query Text: {first_row['text_query']}")
        
        print("\n[Testing JSON Parsing for 'query_result']...")
        try:
            data_sample = json.loads(first_row['query_result'])
            print(f"✅ JSON is valid! Found {len(data_sample)} records in this result.")
            print(f"   First record sample: {data_sample[0]}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parsing Failed: {e}")
            print("   Raw string start:", str(first_row['query_result'])[:100])

        print("\n[Testing Code Column 'visualization_code']...")
        code_sample = first_row['visualization_code']
        if isinstance(code_sample, str) and len(code_sample) > 10:
             print("✅ Code column looks like a string.")
             print("   Snippet:\n", code_sample[:100] + "...")
        else:
             print("❌ Code column seems empty or malformed.")

    except FileNotFoundError:
        print(f"❌ File not found: {OUTPUT_DATASET}")
    except pd.errors.ParserError as e:
        print(f"❌ CSV Parser Error (Structure is broken): {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


class GDELTFullLoader:
    def __init__(self, path_mentions, path_export):
        self.path_mentions = path_mentions
        self.path_export = path_export
        
        # Schemas
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

    def load_full_data(self):
        print("--- Loading Full Events Table ---")
        events = self._load_all_files(
            self.path_export, 
            "*.export.CSV", 
            self.COL_NAMES_EVENTS, 
            {"GlobalEventID": "string", "EventCode": "string"}
        )
        
        print("--- Loading Full Mentions Table ---")
        mentions = self._load_all_files(
            self.path_mentions, 
            "*.mentions.CSV", 
            self.COL_NAMES_MENTIONS, 
            {"GlobalEventID": "string"}
        )
        
        return {"events": events, "mentions": mentions}

    def _load_all_files(self, directory, file_pattern, col_names, dtype_dict):
        search_path = os.path.join(directory, file_pattern)
        files = glob.glob(search_path)[:10]  # Limit to first 10 files for performance
        
        if not files:
            print(f"Warning: No files found in {search_path}")
            return pd.DataFrame(columns=col_names)
        
        print(f"Found {len(files)} files. processing...")
        df_list = []
        for filename in files:
            try:
                if os.path.basename(filename)[0].isdigit():
                    df = pd.read_csv(
                        filename, sep="\t", names=col_names, dtype=dtype_dict, on_bad_lines='skip'
                    )
                    df_list.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            print(f"Loaded {len(final_df)} rows.")
            return final_df
        return pd.DataFrame(columns=col_names)

# ==========================================
# PART 2: LANGCHAIN VIZ PIPELINE
# ==========================================

viz_gen_system = """
You are a Python Data Visualization Expert.
Your task is to generate Python code using `matplotlib` or `seaborn` to visualize a dataset provided by the user.

Input provided:
1. The User's Text Query (Context).
2. The Data (A sample of the pandas DataFrame results).

Requirements:
- The code must assume the data is already loaded into a variable named `df`.
- **DO NOT** read csv files. Use `df`.
- Use `matplotlib.pyplot` or `seaborn`.
- Ensure the plot has a title, labels, and a legend (if applicable).
- The code should be clean and executable.
- Do NOT use `plt.show()` (the execution engine handles display/saving), but ensure the figure is created.
- **SEABORN SPECIFIC**: If using `sns.barplot` (or similar) with a `palette`, you MUST assign `hue` to the same variable as `x` and set `legend=False`. Example: `sns.barplot(x='col', y='val', hue='col', legend=False, palette='viridis')`.
- Output **ONLY** the python code block. No markdown backticks, no explanation.
"""

viz_gen_user = """
User Query: {text_query}

Data Sample (first 5 rows):
{data_head}

Data Columns & Types:
{data_types}

Generate the Python code now.
"""

viz_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", viz_gen_system),
    ("user", viz_gen_user)
])

viz_fix_system = """
You are a Python Debugger.
The previous code failed to execute or produced an empty plot.
Fix the code based on the error message.

Rules:
- Assume data is in `df`.
- Use `matplotlib` or `seaborn`.
- Output **ONLY** the valid python code block.
"""

viz_fix_user = """
Broken Code:
{code}

Error Traceback:
{error}

Please provide the corrected code.
"""

viz_fix_prompt = ChatPromptTemplate.from_messages([
    ("system", viz_fix_system),
    ("user", viz_fix_user)
])

code_gen_chain = viz_gen_prompt | llm | StrOutputParser()
code_fix_chain = viz_fix_prompt | llm | StrOutputParser()

# ==========================================
# PART 3: VALIDATION ENGINE
# ==========================================

def validate_and_execute_viz(code, df):
    """
    Executes the generated code against the dataframe 'df'.
    Returns: (success: bool, output: str or error)
    """
    exec_globals = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'df': df.copy()
    }
    
    f = io.StringIO()
    
    try:
        plt.clf()
        plt.close('all')
        
        with contextlib.redirect_stdout(f):
            exec(code, exec_globals)
            
        if not plt.get_fignums():
            return False, "Code executed but no matplotlib figure was created. Did you forget to plot?"
            
        return True, "Success"
        
    except Exception:
        return False, traceback.format_exc()

def clean_code_string(code_str):
    """Removes markdown backticks if the LLM includes them."""
    cleaned = code_str.replace("```python", "").replace("```", "").strip()
    return cleaned

# ==========================================
# PART 4: MAIN PIPELINE
# ==========================================

def run_viz_pipeline():
    # 1. Check Input
    if not os.path.exists(INPUT_QUERIES_FILE):
        print(f"Error: {INPUT_QUERIES_FILE} not found. Run the previous script first.")
        return

    print("--- Loading Input Queries ---")
    df_queries = pd.read_csv(INPUT_QUERIES_FILE, quotechar='"', skipinitialspace=True)
    print(f"Loaded {len(df_queries)} queries.")

    # 2. Setup Output / Resume Logic
    # We maintain a set of IDs that are already done
    processed_ids = set()
    
    # Column definition for the output
    out_cols = ["id", "text_query", "sql_query", "query_result", "visualization_code"]

    if os.path.exists(OUTPUT_DATASET):
        print(f"Found existing output file: {OUTPUT_DATASET}")
        try:
            # We only need the 'id' column to know what to skip
            existing_df = pd.read_csv(OUTPUT_DATASET, usecols=['id'])
            processed_ids = set(existing_df['id'].unique())
            print(f"Resuming... {len(processed_ids)} queries already processed.")
        except Exception as e:
            print(f"Warning: Could not read existing file ({e}). Starting fresh.")
    else:
        print("Creating new output file.")
        # Create empty file with headers
        pd.DataFrame(columns=out_cols).to_csv(OUTPUT_DATASET, index=False, quoting=1)

    # 3. Load Data
    loader = GDELTFullLoader(PATH_MENTIONS, PATH_EXPORT)
    env = loader.load_full_data()
    
    print("\n--- Starting Execution & Visualization Loop ---")
    
    # 4. Processing Loop
    for index, row in df_queries.iterrows():
        try:
            q_id = row['id']
            sql = row['sql_query']
            text = row['text_query']
            
            # --- SKIP LOGIC ---
            if q_id in processed_ids:
                print(f"Skipping ID {q_id} (Already done)")
                continue

            print(f"\nProcessing Query ID: {q_id}")
            
            # --- SQL Execution ---
            try:
                result_df = sqldf(sql, env)
            except Exception as e:
                print(f"SQL Execution Failed for ID {q_id}: {e}")
                continue 

            if result_df.empty:
                print(f"Query returned empty result. Skipping visualization.")
                continue

            print(f"  > Generating Viz Code for {len(result_df)} rows...")
            
            data_head = result_df.head(5).to_markdown(index=False)
            data_types = str(result_df.dtypes)
            
            generated_code = code_gen_chain.invoke({
                "text_query": text,
                "data_head": data_head,
                "data_types": data_types
            })
            
            generated_code = clean_code_string(generated_code)

            max_retries = 3
            valid_code = None
            
            for attempt in range(max_retries):
                is_valid, message = validate_and_execute_viz(generated_code, result_df)
                
                if is_valid:
                    print(f"  > ✅ Code Validated (Attempt {attempt+1})")
                    valid_code = generated_code
                    break
                else:
                    print(f"  > ❌ Code Failed (Attempt {attempt+1}): {message.splitlines()[-1]}")
                    
                    generated_code = code_fix_chain.invoke({
                        "code": generated_code,
                        "error": message
                    })
                    generated_code = clean_code_string(generated_code)

            if valid_code:
                result_str = result_df.to_json(orient='records')
                
                new_row = pd.DataFrame([{
                    "id": q_id,
                    "text_query": text,
                    "sql_query": sql,
                    "query_result": result_str,
                    "visualization_code": valid_code
                }])
                
                new_row.to_csv(OUTPUT_DATASET, mode='a', header=False, index=False, quoting=1)
                
                print(f"  > Saved ID {q_id} to {OUTPUT_DATASET}")
                
                processed_ids.add(q_id)
            else:
                print(f"  > Failed to generate valid code after {max_retries} attempts.")

        except Exception as e:
            print(f"Critical Error processing row {index}: {e}")

    print("\n--- Pipeline Completed ---")

if __name__ == "__main__":
    reindex_ids()
    test_readability()
    plt.switch_backend('Agg') 
    run_viz_pipeline()
import os
import glob
import random
import json
import pandas as pd
from pandasql import sqldf
from io import StringIO

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# PART 1: THE VALIDATOR ENGINE (Updated)
# ==========================================

class GDELTValidator:
    def __init__(self, path_mentions, path_export):
        self.path_mentions = path_mentions
        self.path_export = path_export
        self.events = None
        self.mentions = None
        self.env = {}
        
        # Schema Definitions
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
        print("--- Loading Events ---")
        self.events = self._load_files(self.path_export, "*.export.CSV", self.COL_NAMES_EVENTS, {"GlobalEventID": "string", "EventCode": "string"})
        print(f"Total Events loaded: {len(self.events)}")

        print("--- Loading Mentions ---")
        self.mentions = self._load_files(self.path_mentions, "*.mentions.CSV", self.COL_NAMES_MENTIONS, {"GlobalEventID": "string"})
        print(f"Total Mentions loaded: {len(self.mentions)}")
        
        self.env = {"events": self.events, "mentions": self.mentions}

    def _load_files(self, directory, file_pattern, col_names, dtype_dict):
        search_path = os.path.join(directory, file_pattern)
        files = glob.glob(search_path)[0:10] # Keeping strict limit for performance
        if not files:
            return pd.DataFrame(columns=col_names)
        
        df_list = []
        for filename in files:
            try:
                # GDELT is tab-separated, no headers
                df = pd.read_csv(filename, sep="\t", names=col_names, dtype=dtype_dict, on_bad_lines='skip')
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if df_list:
            return pd.concat(df_list, ignore_index=True)
        return pd.DataFrame(columns=col_names)

    def execute_query(self, sql_query):
        """
        Runs query. 
        Returns: (Success (bool), Result/Error (str))
        If success, Result is a markdown string of the dataframe head.
        """
        try:
            # Execute SQL
            result_df = sqldf(sql_query, self.env)
            
            # Formatting result for the LLM Judge
            if result_df.empty:
                return True, "Empty Result (0 rows returned)"
            
            # Return first 5 rows as markdown for the judge to inspect
            return True, result_df.head(5).to_markdown(index=False)
            
        except Exception as e:
            return False, str(e)

# ==========================================
# PART 2: THE LLM JUDGE
# ==========================================

class SQLJudge:
    def __init__(self, llm_model, codebook_text):
        self.llm = llm_model
        self.codebook = codebook_text 
        
        self.system_prompt = """
        You are a SQL Quality Assurance Judge specializing in the GDELT Dataset.
        
        DATASET SCHEMA / CODEBOOK:
        {codebook}
        
        You will receive:
        1. A user question (in French).
        2. A generated SQL query.
        3. The execution result of that SQL (or an error message).
        
        Your task is to evaluate if the SQL correctly answers the question AND returns valid data.
        
        SCORING CRITERIA (0.0 to 1.0):
        - 1.0: Perfect. Logic is correct, syntax is correct, AND it returns actual data (non-empty).
        - 0.8: Logically correct, but minor formatting issues. (Must return data).
        - 0.5: Empty Result. Logic might be correct, but the specific filter values found no data.
        - 0.0: Execution Error or completely wrong logic.

        CRITICAL INSTRUCTION ON EMPTY RESULTS:
        - We need queries that generate ACTUAL DATA for a training dataset.
        - If the execution result is "Empty Result" (0 rows), you must REJECT it (Score < 0.8).
        - In the "reasoning", you must specifically instruct to CHANGE THE FILTER VALUES.
          (e.g., "The query is logically correct but returned no data. Try changing the CountryCode from 'ATA' (Antarctica) to something common like 'US' or 'FR', or broaden the date range.")
          DON'T FORGET TO CHANGE THE QUESTION IN FRENCH TOO to match the new filters!
        
        Output valid JSON only:
        {{
            "score": float,
            "reasoning": "string explanation of why valid or invalid"
        }}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """
            QUESTION (FR): {question}
            SQL: {sql}
            EXECUTION RESULT: 
            {result}
            """)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def evaluate(self, question, sql, execution_result):
        try:
            response = self.chain.invoke({
                "codebook": self.codebook,  # Pass the codebook context here
                "question": question,
                "sql": sql,
                "result": execution_result
            })
            
            # Clean response to ensure JSON parsing
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Judge Error: {e}")
            return {"score": 0.0, "reasoning": "Judge failed to parse response."}

# ==========================================
# PART 3: GENERATION & FIX SETUP
# ==========================================

# Initialize LLM
llm = ChatOpenAI(
    model="deepseek-chat",              
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0,
    max_tokens=8000
)

# 1. GENERATION PROMPT

gen_system_prompt = """You are a SQL expert specializing in the GDELT 2.0 Dataset.

Your goal is to generate unique, analytical SQL queries based on the provided GDELT Codebook.

Output must be a valid CSV format with columns: id, text_query, sql_query.

Use a semicolon (,) as the CSV separator to avoid conflicts with SQL commas.
Quote all fields with double quotes.
Write text_query in French. Also, make the text_query as simple and clear as possible (don't use the name of columns in the text_query).

Dataset Rules:
1. Tables are named 'events' and 'mentions'.
2. Join them on 'GlobalEventID'.
3. Use standard SQLite syntax (supported by pandasql).
4. Do NOT use functions like STDDEV (use simple math or approximations).
5. Use descriptive French aliases for all selected columns (e.g., AS 'Date_Evenement').


GDELT Schema:
{codebook}

Example Valid Queries:
{examples}

So make sure to seperate columns with , and not ;.
Also, make sure to quote all fields with double quotes.

generate {n} distinct, complex SQL queries (starting ID: {start_id}).

```csv
"id", "text_query", "sql_query"
"""

gen_user_prompt = "Generate {n} distinct, complex SQL queries (starting ID: {start_id})."



gen_prompt = ChatPromptTemplate.from_messages([

    ("system", gen_system_prompt),
    ("user", gen_user_prompt)
])
gen_chain = gen_prompt | llm | StrOutputParser()

# 2. REPAIR AGENT (Fixes Syntax AND Logic)
fix_system_prompt = """
You are a SQL Debugging Assistant.
You will receive a CSV row (id, text_query, sql_query) and specific FEEDBACK.

The Feedback can be:
1. An Execution Error (Python/SQLite error).
2. A Logic critique from a QA Judge (e.g., "The query doesn't match the question or use wrong conditions").

Your job: Rewrite the SQL query and/or the text_query to satisfy the feedback.
Output ONLY the corrected CSV row using comma separators.
```csv
"id", "text_query", "sql_query"
"""

fix_prompt = ChatPromptTemplate.from_messages([
    ("system", fix_system_prompt),
    ("user", "Row: {row}\nFeedback: {feedback}")
])
fix_chain = fix_prompt | llm | StrOutputParser()

# ==========================================
# PART 4: ORCHESTRATION
# ==========================================

def run_pipeline():
    # 1. CONFIG
    PATH_MENTIONS = "./gdelt_data_event_mentions/" 
    PATH_EXPORT = "./gdelt_data_event_export/"
    N_QUERIES = 20  
    JUDGE_THRESHOLD = 1.0 

    # 2. LOAD DATA
    validator = GDELTValidator(PATH_MENTIONS, PATH_EXPORT)
    validator.load_data()
    

    # 3. GET CONTEXT (Simulated here)
    try:
        with open("./GDELT_import_and_codebook/gdelt_schema.txt", "r") as f:
            codebook = f.read()
    except: codebook = "Table events, Table mentions..."
    
    with open("./GDELT_import_and_codebook/ttsql_training_dataset_GDELT.csv", "r", encoding="utf-8") as f:

        list_examples = f.readlines()
        example_queries_text = ""
        list_indexes = []
        for i in range (4):
            index = random.randint(0, len(list_examples)-1)
            while index in list_indexes:
                index = random.randint(0, len(list_examples)-1)
            list_indexes.append(index)
            example_queries_text += list_examples[list_indexes[i]] 
    
    examples = example_queries_text 

    # 4. SETUP JUDGE
    judge = SQLJudge(llm, codebook)
    
    # 5. GENERATE
    print(f"\n--- Generating {N_QUERIES} Queries ---")
    csv_output = gen_chain.invoke({
        "codebook": codebook,
        "examples": examples,
        "n": N_QUERIES,
        "start_id": 1
    })
    
    # Parse generated CSV
    try:
        df_queries = pd.read_csv(StringIO(csv_output), header=None, names=["id", "text_query", "sql_query"], quotechar='"', skipinitialspace=True)
    except Exception as e:
        print("CSV Parsing failed on generation output. Raw output:")
        print(csv_output)
        return

    final_rows = []

    # 6. VALIDATION LOOP
    print("\n--- Validating & Judging ---")
    
    for index, row in df_queries.iterrows():
        q_id, text, sql = row['id'], row['text_query'], row['sql_query']
        print(f"\nProcessing ID {q_id}...")

        # We allow a few attempts to fix either syntax or logic
        attempts = 0
        max_attempts = 2
        is_accepted = False
        current_sql = sql
        
        while attempts <= max_attempts and not is_accepted:
            # A. Execution Check
            success, exec_result = validator.execute_query(current_sql)
            
            if not success:
                # FAILURE: Syntax Error
                print(f"  ❌ Syntax Error: {exec_result}")
                feedback = f"SQLite Error: {exec_result}"
                needs_fix = True
            else:
                # SUCCESS: Syntax is good, now check Logic
                print("  ✅ Execution Valid. Running Judge...")
                judgment = judge.evaluate(text, current_sql, exec_result)
                score = judgment.get('score', 0)
                reasoning = judgment.get('reasoning', 'No reasoning')
                
                print(f"  ⚖️  Judge Score: {score}/1.0 | Reason: {reasoning}")
                
                if score >= JUDGE_THRESHOLD:
                    is_accepted = True
                    needs_fix = False
                else:
                    feedback = f"Logic Error. Judge Reasoning: {reasoning}"
                    needs_fix = True

            # B. Fix Step (if needed)
            if needs_fix and attempts < max_attempts:
                print("  🔧 Attempting Fix...")
                try:
                    csv_line = f'"{q_id}","{text}","{current_sql}"'
                    fixed_csv = fix_chain.invoke({"row": csv_line, "feedback": feedback})
                    
                    # Parse fix
                    fixed_df = pd.read_csv(StringIO(fixed_csv), header=None, names=["id", "text_query", "sql_query"], quotechar='"')
                    current_sql = fixed_df.iloc[0]['sql_query']
                except Exception as e:
                    print(f"  ❌ Repair Failed: {e}")
                    break
            
            attempts += 1

        if is_accepted:
            # Update the row with the potentially fixed SQL
            row['sql_query'] = current_sql
            final_rows.append(row)
            print("  🎉 Query Accepted!")
        else:
            print("  🗑️  Query Discarded.")

    # 7. EXPORT
    if final_rows:
        new_df = pd.DataFrame(final_rows)
        output_path = "./validated_gdelt_queries.csv"
        
        file_exists = os.path.isfile(output_path)
        
        new_df.to_csv(
            output_path, 
            mode='a', 
            index=False, 
            quoting=1, 
            header=not file_exists
        )
        
        status = "Appended to" if file_exists else "Created"
        print(f"\n{status} {output_path} with {len(new_df)} new validated queries.") 
if __name__ == "__main__":
    for i in range (100):
        run_pipeline()
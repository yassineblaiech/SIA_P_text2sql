import os
import glob
import random
import json
import pandas as pd
from pandasql import sqldf
from io import StringIO
import csv
import sys
import time  # <-- Required for the retry sleep

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# PART 0: RETRY UTILITY
# ==========================================

def invoke_with_retry(chain, input_data, max_retries=10):
    """Executes a LangChain chain with a strict retry limit."""
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            print(e)
            print(f"    ⚠️ [Attempt {attempt}/{max_retries}] LLM timeout or error: {e}. Retrying...")
            if attempt == max_retries:
                print("    ❌ Max retries reached (10). Discarding task.")
                raise e 
            time.sleep(1)

# ==========================================
# PART 1: THE VALIDATOR ENGINE
# ==========================================

class BudgetDataValidator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.env = {}
        
        # Schema Definitions (Columns)
        self.SCHEMAS = {
            "documents": ["document_id", "mask", "siret", "city_name", "format", "document_year", "created_at", "link_bucketS3", "state_id", "error_message", "warning_message", "budget_type", "nb_pages"],
            "extraction_runs": ["run_id", "document_id", "start_time", "config", "state_id"],
            "extraction_results": ["result_id", "safir_field_id", "run_id", "document_id", "value", "created_at", "message_error", "message_warning", "state_id"],
            "resultfeedbacks": ["feedback_id", "result_id", "is_technical_error", "comment", "corrected_value", "created_at"],
            "states": ["state_id", "code"],
            "sections": ["section_id", "document_id", "created_at", "section_value"],
            "rulesets": ["ruleset_id", "description", "created_at"],
            "field_rules": ["extraction_rule_id", "safir_field_id", "mask", "definition", "ruleset_id"],
            "validation_rules": ["validation_rule_id", "definition", "level_of_problem", "mask", "ruleset_id", "message_if_problem", "safir_field_id_list"]
        }

    def load_data(self):
        print("--- Loading Budget Tables ---")
        for table_name, cols in self.SCHEMAS.items():
            df = self._load_file(table_name, cols)
            self.env[table_name] = df
            print(f"Table '{table_name}' loaded: {len(df)} rows")

    def _load_file(self, table_name, col_names):
        search_path = os.path.join(self.data_dir, f"{table_name}.csv")
        if not os.path.exists(search_path):
            return pd.DataFrame(columns=col_names)
        
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2147483647)
        
        try:
            cleaned_lines = []
            with open(search_path, 'r', encoding='utf-8') as f:
                for line in f:
                    cleaned_line = line.replace('\'"', '""')
                    cleaned_lines.append(cleaned_line)
            
            if not cleaned_lines:
                return pd.DataFrame(columns=col_names)

            if col_names[0] in cleaned_lines[0]:
                cleaned_lines.pop(0) 
                
            reader = csv.reader(cleaned_lines, delimiter=',', quotechar='"', escapechar='\\')
            parsed_data = []
            
            for row in reader:
                if not row: 
                    continue
                
                if len(row) < len(col_names):
                    row.extend([None] * (len(col_names) - len(row)))
                elif len(row) > len(col_names):
                    row = row[:len(col_names)]
                    
                parsed_data.append(row)
            
            df = pd.DataFrame(parsed_data, columns=col_names)
            return df

        except Exception as e:
            print(f"Error loading {search_path}: {e}")
            return pd.DataFrame(columns=col_names)

    def execute_query(self, sql_query):
        try:
            result_df = sqldf(sql_query, self.env)
            if result_df is None or result_df.empty:
                return True, "Empty Result (0 rows returned)"
            return True, result_df.head(5).to_markdown(index=False)
        except Exception as e:
            return False, str(e)
        
    def print_all_heads(self, n=5):
        print(f"\n--- Checking Table Heads (Top {n} rows) ---")
        for table_name, df in self.env.items():
            if table_name == 'sections':
                continue
            print(f"\n=== Table: {table_name} ===")
            if df.empty:
                print("(Table is empty / No data loaded)")
            else:
                print(df.head(n).to_markdown(index=False))

# ==========================================
# PART 2: THE LLM JUDGE
# ==========================================

class SQLJudge:
    def __init__(self, llm_model, codebook_text):
        self.llm = llm_model
        self.codebook = codebook_text 
        
        self.system_prompt = """
        You are a SQL Quality Assurance Judge specializing in a Budget Document Extraction Database.
        
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
          (e.g., "The query is logically correct but returned no data. Try changing the budget_type from 'BA' to 'BP' or adjust the safir_field_id.")
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
            # Replaced standard invoke with retry utility
            response = invoke_with_retry(self.chain, {
                "codebook": self.codebook,
                "question": question,
                "sql": sql,
                "result": execution_result
            })
            
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"    ❌ Judge completely failed after 10 retries.")
            return {"score": 0.0, "reasoning": "Judge failed or timed out repeatedly."}

# ==========================================
# PART 3: GENERATION & FIX SETUP
# ==========================================

llm = ChatOpenAI(
    model="gpt-oss-120b", 
    base_url="https://api.cerebras.ai/v1", 
    api_key=os.environ.get("CEREBRAS_API_KEY"),
    temperature=0,
    max_tokens=8000,
    timeout=2000.0,    
    max_retries=0   
)

# GENERATION PROMPT
gen_system_prompt = """You are a SQL expert specializing in a Budget Document Extraction Database.

Your goal is to generate unique, analytical SQL queries based on the provided Budget Database Codebook.

Output must be a valid CSV format with columns: id, text_query, sql_query.

Use a semicolon (,) as the CSV separator to avoid conflicts with SQL commas.
Quote all fields with double quotes.
Write text_query in French. Also, make the text_query as simple and clear as possible (don't use the raw names of columns like 'safir_field_id' in the text_query, say "l'identifiant fonctionnel du champ").

Dataset Rules:
1. The core tables are 'documents', 'extraction_results', 'extraction_runs', 'states', 'rulesets', 'field_rules', etc.
2. Join properly (e.g., documents.document_id = extraction_results.document_id).
3. Use standard SQLite syntax (supported by pandasql).
4. Use descriptive French aliases for all selected columns (e.g., AS 'Type_Budget').

Budget Schema Context:
{codebook}

Example Valid Queries:
{examples}

Make sure to separate columns with , and not ;.
Make sure to quote all fields with double quotes.

Generate {n} distinct, complex SQL queries (starting ID: {start_id}).

```csv
"id", "text_query", "sql_query"
"""
gen_user_prompt = "Generate {n} distinct, complex SQL queries (starting ID: {start_id})."

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", gen_system_prompt),
    ("user", gen_user_prompt)
])
gen_chain = gen_prompt | llm | StrOutputParser()

# REPAIR AGENT
fix_system_prompt = """
You are a SQL Debugging Assistant.
You will receive a CSV row (id, text_query, sql_query) and specific FEEDBACK.

The Feedback can be:
1. An Execution Error (Python/SQLite error).
2. A Logic critique from a QA Judge (e.g., "The query returned empty results, change the filter values").

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
    PATH_DATA = "./SIA_db/pgadmin_exports" 
    N_QUERIES = 20  
    JUDGE_THRESHOLD = 1.0 

    # 2. LOAD DATA
    validator = BudgetDataValidator(PATH_DATA)
    validator.load_data()
    
    # 3. GET CONTEXT
    with open("./SIA_db/context.txt", "r", encoding="utf-8") as f:
        codebook = f.read()
        
    with open("./SIA_db/budget_queries_examples.csv", "r", encoding="utf-8") as f:
        list_examples = f.readlines()
        example_queries_text = ""
        list_indexes = []
        for i in range(4):
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
    
    try:
        csv_output = invoke_with_retry(gen_chain, {
            "codebook": codebook,
            "examples": examples,
            "n": N_QUERIES,
            "start_id": 1
        })
    except Exception:
        print("  ⏭️ Skipping this generation batch due to repeated timeouts.")
        return

    # Clean the Generation Output
    clean_csv = csv_output.replace("```csv", "").replace("```", "").strip()
    
    valid_lines = []
    for line in clean_csv.split('\n'):
        line_lower = line.strip().lower()
        if not line_lower:
            continue
        # Skip the header line
        if "id" in line_lower and "text_query" in line_lower and "sql_query" in line_lower:
            continue
        valid_lines.append(line.strip())
        
    final_csv_string = "\n".join(valid_lines)
    
    # Parse generated CSV
    try:
        df_queries = pd.read_csv(StringIO(final_csv_string), header=None, names=["id", "text_query", "sql_query"], quotechar='"', skipinitialspace=True)
    except Exception as e:
        print("CSV Parsing failed on generation output. Raw output:")
        print(csv_output)
        print("\nCleaned output attempted:")
        print(final_csv_string)
        return

    final_rows = []

    # 6. VALIDATION LOOP
    print("\n--- Validating & Judging ---")
    
    for index, row in df_queries.iterrows():
        q_id, text, sql = row['id'], row['text_query'], row['sql_query']
        print(f"\nProcessing ID {q_id}...")

        attempts = 0
        max_attempts = 2
        is_accepted = False
        current_sql = sql
        current_text = text
        
        while attempts <= max_attempts and not is_accepted:
            # A. Execution Check
            success, exec_result = validator.execute_query(current_sql)
            
            if not success:
                print(f"  ❌ Syntax Error: {exec_result}")
                feedback = f"SQLite Error: {exec_result}"
                needs_fix = True
            else:
                print("  ✅ Execution Valid. Running Judge...")
                judgment = judge.evaluate(current_text, current_sql, exec_result)
                score = judgment.get('score', 0)
                reasoning = judgment.get('reasoning', 'No reasoning')
                
                print(f"  ⚖️  Judge Score: {score}/1.0 | Reason: {reasoning}")
                
                if score >= JUDGE_THRESHOLD:
                    is_accepted = True
                    needs_fix = False
                else:
                    feedback = f"Logic Error. Judge Reasoning: {reasoning}"
                    needs_fix = True

            # B. Fix Step
            if needs_fix and attempts < max_attempts:
                print("  🔧 Attempting Fix...")
                try:
                    csv_line = f'"{q_id}","{current_text}","{current_sql}"'
                    
                    fixed_csv = invoke_with_retry(fix_chain, {"row": csv_line, "feedback": feedback})
                    
                    # Clean the output from the Fix Agent as well
                    clean_fixed = fixed_csv.replace("```csv", "").replace("```", "").strip()
                    fixed_valid = []
                    for line in clean_fixed.split('\n'):
                        line_lower = line.strip().lower()
                        if not line_lower or ("id" in line_lower and "text_query" in line_lower):
                            continue
                        fixed_valid.append(line.strip())
                    
                    final_fixed_string = "\n".join(fixed_valid)
                    
                    fixed_df = pd.read_csv(StringIO(final_fixed_string), header=None, names=["id", "text_query", "sql_query"], quotechar='"')
                    current_text = fixed_df.iloc[0]['text_query']
                    current_sql = fixed_df.iloc[0]['sql_query']
                except Exception as e:
                    print(f"  ❌ Repair Failed or Timed Out completely.")
                    break
            
            attempts += 1

        if is_accepted:
            row['sql_query'] = current_sql
            row['text_query'] = current_text
            final_rows.append(row)
            print("  🎉 Query Accepted!")
        else:
            print("  🗑️  Query Discarded.")

    # 7. EXPORT
    if final_rows:
        new_df = pd.DataFrame(final_rows)
        output_path = "./SIA_db/gpt_oss_queries.csv"
        
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
    for i in range(50):
        run_pipeline()
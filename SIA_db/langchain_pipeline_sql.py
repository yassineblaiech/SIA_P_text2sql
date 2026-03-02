import os
import glob
import random
import json
import pandas as pd
from pandasql import sqldf
from io import StringIO
import csv
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# PART 1: THE VALIDATOR ENGINE (Adapted for Budget Data)
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
            # Assumes CSV files are named like "documents.csv", "states.csv", etc.
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
            # Fallback de sécurité pour les systèmes Windows 64-bit
            csv.field_size_limit(2147483647)
        
        try:
                cleaned_lines = []
            
                # 1. Lecture et nettoyage brut du texte
                with open(search_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # On corrige le bug des guillemets pgAdmin dans le JSON
                        cleaned_line = line.replace('\'"', '""')
                        cleaned_lines.append(cleaned_line)
                
                if not cleaned_lines:
                    return pd.DataFrame(columns=col_names)

                # 2. Détection intelligente de l'en-tête
                # Si le premier nom de colonne se trouve dans la première ligne, c'est qu'il y a un header
                if col_names[0] in cleaned_lines[0]:
                    cleaned_lines.pop(0) # On le retire pour ne garder que les données
                    
                # 3. Parsing robuste avec le module natif Python
                # Il gère beaucoup mieux les blocs de texte complexes que Pandas
                reader = csv.reader(cleaned_lines, delimiter=',', quotechar='"', escapechar='\\')
                
                parsed_data = []
                for row in reader:
                    if not row: # Ignore les lignes totalement vides
                        continue
                    
                    # 4. Forçage strict de la structure
                    # S'il manque des colonnes (ex: champs vides à la fin), on complète avec du vide (None)
                    if len(row) < len(col_names):
                        row.extend([None] * (len(col_names) - len(row)))
                    # S'il y a trop de colonnes (ex: virgule non échappée), on tronque pour éviter de crasher
                    elif len(row) > len(col_names):
                        row = row[:len(col_names)]
                        
                    parsed_data.append(row)
                
                # 5. Création finale de la DataFrame propre
                df = pd.DataFrame(parsed_data, columns=col_names)
                return df

                    
            
        except Exception as e:
            print(f"Error loading {search_path}: {e}")
            return pd.DataFrame(columns=col_names)

    def execute_query(self, sql_query):
        """
        Runs query. 
        Returns: (Success (bool), Result/Error (str))
        """
        try:
            # Execute SQL
            result_df = sqldf(sql_query, self.env)
            
            # Formatting result for the LLM Judge
            if result_df is None or result_df.empty:
                return True, "Empty Result (0 rows returned)"
            
            # Return first 5 rows as markdown for the judge to inspect
            return True, result_df.head(5).to_markdown(index=False)
            
        except Exception as e:
            return False, str(e)
        
    def print_all_heads(self, n=5):
        """
        Prints the first n rows of all loaded tables in the environment
        to verify that columns and data are formatted correctly.
        """
        print(f"\n--- Checking Table Heads (Top {n} rows) ---")
        for table_name, df in self.env.items():
            if table_name == 'sections':
                continue
            print(f"\n=== Table: {table_name} ===")
            if df.empty:
                print("(Table is empty / No data loaded)")
            else:
                # Using to_markdown for a cleaner console output
                print(df.head(n).to_markdown(index=False))

# ==========================================
# PART 2: THE LLM JUDGE (Adapted for Budget Data)
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
            response = self.chain.invoke({
                "codebook": self.codebook,
                "question": question,
                "sql": sql,
                "result": execution_result
            })
            
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Judge Error: {e}")
            return {"score": 0.0, "reasoning": "Judge failed to parse response."}

# ==========================================
# PART 3: GENERATION & FIX SETUP (Adapted)
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

# 2. REPAIR AGENT
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
                    fixed_csv = fix_chain.invoke({"row": csv_line, "feedback": feedback})
                    
                    fixed_df = pd.read_csv(StringIO(fixed_csv), header=None, names=["id", "text_query", "sql_query"], quotechar='"')
                    current_text = fixed_df.iloc[0]['text_query']
                    current_sql = fixed_df.iloc[0]['sql_query']
                except Exception as e:
                    print(f"  ❌ Repair Failed: {e}")
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
        output_path = "./SIA_db/validated_budget_queries.csv"
        
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
    # PATH_DATA = "./SIA_db/pgadmin_exports" 
    # validator = BudgetDataValidator(PATH_DATA)
    # validator.load_data()
    # validator.print_all_heads()
    for i in range(1):
        run_pipeline()
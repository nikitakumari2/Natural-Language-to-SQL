# import os
# from dotenv import load_dotenv
# import json
# import time
# from datetime import datetime
# from pathlib import Path
# import pandas as pd

# from langchain import HuggingFaceHub
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import VertexAI, HuggingFacePipeline

# # Import our SQL copilot functions
# from sql_functions import (
#     similar_doc_search, identify_schemas, connect_db, prioritize_tables,
#     get_table_info, get_sql_dialect, llm_create_sql, llm_check_sql,
#     run_sql, llm_debug_error, llm_debug_empty, llm_analyze
# )

# # For local HF models
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# # Setup embeddings using HuggingFace and the directory location
# embeddings = HuggingFaceEmbeddings()
# persist_dir = '../data/processed/chromadb'
# db_filepath = '../data/raw/spider/database/'

# # Load from disk
# vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# # Get API keys and tokens
# load_dotenv()
# hf_api_token = os.getenv('hf_token')

# class ModelComparison:
#     """Class to manage and run comparisons between different LLMs for SQL generation."""
    
#     def __init__(self, results_dir='../results'):
#         """Initialize the model comparison system.
        
#         Args:
#             results_dir: Directory to save comparison results
#         """
#         self.results_dir = Path(results_dir)
#         self.results_dir.mkdir(exist_ok=True, parents=True)
#         self.models = {}
#         self._initialize_default_models()
        
#     def _initialize_default_models(self):
#         """Initialize the default set of models for comparison."""
#         # OpenAI GPT-3.5
#         self.add_model(
#             "gpt-3.5-turbo", 
#             ChatOpenAI(
#                 model_name="gpt-3.5-turbo",
#                 temperature=0.05,
#                 openai_api_key=os.getenv("OPENAI_API_KEY")
#             )
#         )
        
#         # Add OpenAI GPT-4 if API key is available
#         if os.getenv("OPENAI_API_KEY"):
#             self.add_model(
#                 "gpt-4", 
#                 ChatOpenAI(
#                     model_name="gpt-4",
#                     temperature=0.05,
#                     openai_api_key=os.getenv("OPENAI_API_KEY")
#                 )
#             )
        
#         # Add HuggingFace hub models if token is available
#         if hf_api_token:
#             # Mixtral
#             self.add_model(
#                 "mixtral-8x7b", 
#                 HuggingFaceHub(
#                     repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
#                     model_kwargs={"temperature": 0.05},
#                     huggingfacehub_api_token=hf_api_token
#                 )
#             )
            
#             # FLAN-T5
#             self.add_model(
#                 "flan-t5-xl", 
#                 HuggingFaceHub(
#                     repo_id="google/flan-t5-xl",
#                     model_kwargs={"temperature": 0.05},
#                     huggingfacehub_api_token=hf_api_token
#                 )
#             )

#         # Try to add a local HF model if available
#         try:
#             model_name = "google/flan-t5-base"  # A smaller model that might run locally
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
#             pipe = pipeline(
#                 "text2text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 max_length=512
#             )
            
#             local_llm = HuggingFacePipeline(pipeline=pipe)
#             self.add_model("local-flan-t5", local_llm)
#         except Exception as e:
#             print(f"Could not load local model: {e}")
            
#     def add_model(self, model_name, model_instance):
#         """Add a model to the comparison set.
        
#         Args:
#             model_name: Unique identifier for the model
#             model_instance: LangChain compatible model instance
#         """
#         self.models[model_name] = model_instance
#         print(f"Added model: {model_name}")
        
#     def remove_model(self, model_name):
#         """Remove a model from the comparison set."""
#         if model_name in self.models:
#             del self.models[model_name]
#             print(f"Removed model: {model_name}")
#         else:
#             print(f"Model {model_name} not found")
            
#     def list_models(self):
#         """List all available models for comparison."""
#         return list(self.models.keys())
        
#     def sql_copilot(self, user_question, model_name, max_attempts=3):
#         """Run the SQL copilot with a specific model.
        
#         Args:
#             user_question: The user's natural language query
#             model_name: Name of the model to use
#             max_attempts: Maximum number of attempts to debug SQL
            
#         Returns:
#             dict: Results including SQL, execution time, and answer
#         """
#         if model_name not in self.models:
#             return {
#                 "status": "error",
#                 "message": f"Model {model_name} not found"
#             }
            
#         language_model = self.models[model_name]
#         start_time = time.time()
        
#         print(f"\n--- Using model: {model_name} ---")
#         print("\nIdentifying most likely schemas...")
#         db_documents = similar_doc_search(question=user_question, vector_database=vectordb, top_k=3)

#         top_schemas = identify_schemas(db_documents)
#         print(top_schemas)
#         result = None
        
#         for schema in list(top_schemas):
#             print(f"\nConnecting to {schema} schema...")
#             db = connect_db(db_path=db_filepath, target_schema=schema)
#             print("...Connected to database.")
#             print(db.get_usable_table_names())

#             tables_sorted = prioritize_tables(documents=db_documents, target_schema=schema, sql_database=db)
#             tables_info = get_table_info(tables=tables_sorted, database=db)
#             sql_dialect = get_sql_dialect(database=db)

#             print("\nCalling to language model...")
#             try:
#                 sql_statement = llm_create_sql(
#                     sql_dialect=sql_dialect, 
#                     table_info=tables_info, 
#                     question=user_question, 
#                     lang_model=language_model
#                 )
#                 print(f"\nTry this SQL statement: {sql_statement}")
#             except ValueError as err_msg:
#                 print(f"\n{str(err_msg)}")
#                 print('\nMoving on to try the next schema...')
#                 result = 'FAIL'
#                 continue

#             print("\nValidating SQL...")
#             try:
#                 validated_sql = llm_check_sql(
#                     sql_query=sql_statement, 
#                     sql_dialect=sql_dialect, 
#                     lang_model=language_model
#                 )
#                 print("...SQL validated.")
#             except Exception as e:
#                 print(f"SQL validation failed: {e}")
#                 validated_sql = sql_statement  # Fallback to original statement

#             attempt = 1
#             query_to_run = validated_sql
#             print("\nRunning query on database...")
#             while attempt <= max_attempts:
#                 print(f"Attempting query: {query_to_run}")
#                 query_result = run_sql(database=db, sql_query=query_to_run)

#                 if query_result[:5] == 'Error':
#                     if attempt >= max_attempts:
#                         result = 'FAIL'
#                         output = {
#                             "status": "error",
#                             "message": f"Unable to execute the SQL query after {max_attempts} attempts.",
#                             "final_query": query_to_run,
#                             "error": query_result
#                         }
#                         break

#                     print("\nThat query returned an error. Working on debugging...")
#                     try:
#                         query_to_run = llm_debug_error(
#                             sql_query=query_to_run, 
#                             error_message=query_result, 
#                             lang_model=language_model
#                         )
#                     except Exception as e:
#                         print(f"Error during debugging: {e}")
#                         result = 'FAIL'
#                         output = {
#                             "status": "error",
#                             "message": f"Error during SQL debugging: {str(e)}",
#                             "final_query": query_to_run
#                         }
#                         break
                        
#                     attempt += 1
#                     time.sleep(1)
#                     print("\nTrying again...")

#                 elif query_result == '[]':
#                     if attempt >= max_attempts:
#                         result = 'FAIL'
#                         output = {
#                             "status": "empty_result",
#                             "message": f"Query returned blank results after {max_attempts} attempts.",
#                             "final_query": query_to_run
#                         }
#                         break

#                     print("\nThat query returned no results. Working on debugging...")
#                     try:
#                         query_to_run = llm_debug_empty(
#                             sql_query=query_to_run, 
#                             quesiton=user_question, 
#                             lang_model=language_model
#                         )
#                     except Exception as e:
#                         print(f"Error during empty result debugging: {e}")
#                         result = 'FAIL'
#                         output = {
#                             "status": "error",
#                             "message": f"Error during empty result debugging: {str(e)}",
#                             "final_query": query_to_run
#                         }
#                         break
                        
#                     attempt += 1
#                     time.sleep(1)
#                     print("\nTrying again...")

#                 else:
#                     result = 'SUCCESS'
#                     try:
#                         llm_answer = llm_analyze(
#                             query_result=query_result, 
#                             question=user_question, 
#                             lang_model=language_model
#                         )
#                     except Exception as e:
#                         llm_answer = f"Error analyzing results: {str(e)}"

#                     output = {
#                         "status": "success",
#                         "input_question": user_question,
#                         "original_sql": sql_statement,
#                         "final_sql": query_to_run,
#                         "sql_output": query_result,
#                         "answer": llm_answer,
#                         "schema": schema,
#                         "tables_used": tables_sorted
#                     }
#                     break

#             if result == 'SUCCESS':
#                 break

#         if result != 'SUCCESS':
#             if 'output' not in locals():
#                 output = {
#                     "status": "error",
#                     "message": "Failed to find relevant schema or generate valid SQL"
#                 }
                
#         # Add timing information
#         execution_time = time.time() - start_time
#         output["execution_time"] = execution_time
#         output["model"] = model_name
        
#         return output
        
#     def run_comparison(self, user_question, models_to_use=None, max_attempts=3):
#         """Run a comparison of the SQL copilot across multiple models.
        
#         Args:
#             user_question: The user's natural language query
#             models_to_use: List of model names to use (defaults to all)
#             max_attempts: Maximum SQL debugging attempts per model
            
#         Returns:
#             dict: Comparison results for all models
#         """
#         if not models_to_use:
#             models_to_use = self.list_models()
        
#         results = {}
#         for model_name in models_to_use:
#             print(f"\n{'='*50}")
#             print(f"Running query with model: {model_name}")
#             print(f"{'='*50}")
            
#             try:
#                 model_result = self.sql_copilot(user_question, model_name, max_attempts)
#                 results[model_name] = model_result
#             except Exception as e:
#                 results[model_name] = {
#                     "status": "error",
#                     "message": f"Exception occurred: {str(e)}",
#                     "model": model_name
#                 }
                
#         # Save the comparison results
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = self.results_dir / f"comparison_{timestamp}.json"
        
#         with open(filename, 'w') as f:
#             json.dump({
#                 "question": user_question,
#                 "timestamp": timestamp,
#                 "results": results
#             }, f, indent=2)
            
#         print(f"\nResults saved to {filename}")
#         return results
        
#     def compare_results(self, results):
#         """Generate a summary comparison of results from different models.
        
#         Args:
#             results: Dict with model results from run_comparison
            
#         Returns:
#             pd.DataFrame: Comparison summary
#         """
#         summary = []
        
#         for model_name, result in results.items():
#             row = {
#                 "model": model_name,
#                 "status": result.get("status", "unknown"),
#                 "execution_time": round(result.get("execution_time", 0), 2),
#                 "sql_generated": "original_sql" in result,
#                 "sql_executed": result.get("status") == "success",
#                 "answer_generated": "answer" in result and result["answer"] not in (None, ""),
#             }
#             summary.append(row)
            
#         df = pd.DataFrame(summary)
#         return df
        
#     def analyze_previous_comparisons(self, n=5):
#         """Analyze the most recent comparison files.
        
#         Args:
#             n: Number of most recent comparison files to analyze
            
#         Returns:
#             pd.DataFrame: Aggregated stats across comparisons
#         """
#         files = sorted(self.results_dir.glob("comparison_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
#         files = files[:n]  # Take n most recent
        
#         if not files:
#             return "No comparison files found"
            
#         all_stats = []
        
#         for file in files:
#             with open(file, 'r') as f:
#                 data = json.load(f)
                
#             question = data["question"]
#             timestamp = data["timestamp"]
            
#             for model_name, result in data["results"].items():
#                 row = {
#                     "question": question,
#                     "timestamp": timestamp,
#                     "model": model_name,
#                     "status": result.get("status", "unknown"),
#                     "execution_time": round(result.get("execution_time", 0), 2),
#                     "success": result.get("status") == "success"
#                 }
#                 all_stats.append(row)
                
#         df = pd.DataFrame(all_stats)
#         return df
        
# def main():
#     """Interactive testing of the model comparison system."""
#     comparator = ModelComparison()
    
#     print("\nAvailable models for comparison:")
#     for model in comparator.list_models():
#         print(f"- {model}")
        
#     question = input("\nWhat would you like to know from your data?: ")
    
#     print("\nRunning comparison across all models...")
#     results = comparator.run_comparison(question)
    
#     print("\nComparison Summary:")
#     summary_df = comparator.compare_results(results)
#     print(summary_df)
    
#     # Print the model answers
#     print("\nModel Answers:")
#     for model_name, result in results.items():
#         print(f"\n--- {model_name} ---")
#         if result.get("status") == "success":
#             print(f"Answer: {result.get('answer', 'No answer generated')}")
#         else:
#             print(f"Status: {result.get('status')} - {result.get('message', '')}")

# if __name__ == '__main__':
#     main()

import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import VertexAI, HuggingFacePipeline

# Import our custom HuggingFace wrapper
from custom_hf_wrapper import create_hf_model

# Import our SQL copilot functions
from sql_functions import (
    similar_doc_search, identify_schemas, connect_db, prioritize_tables,
    get_table_info, get_sql_dialect, llm_create_sql, llm_check_sql,
    run_sql, llm_debug_error, llm_debug_empty, llm_analyze
)

# For local HF models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Setup embeddings using HuggingFace and the directory location
embeddings = HuggingFaceEmbeddings()
persist_dir = '../data/processed/chromadb'
db_filepath = '../data/raw/spider/database/'

# Load from disk
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Get API keys and tokens
load_dotenv()
hf_api_token = os.getenv('hf_token')

class ModelComparison:
    """Class to manage and run comparisons between different LLMs for SQL generation."""
    
    def __init__(self, results_dir='../results'):
        """Initialize the model comparison system.
        
        Args:
            results_dir: Directory to save comparison results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models = {}
        self._initialize_default_models()
        
    def _initialize_default_models(self):
        """Initialize the default set of models for comparison."""
        # OpenAI GPT-3.5
        self.add_model(
            "gpt-3.5-turbo", 
            ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.05,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        )
        
        # Add OpenAI GPT-4 if API key is available
        if os.getenv("OPENAI_API_KEY"):
            self.add_model(
                "gpt-4", 
                ChatOpenAI(
                    model_name="gpt-4",
                    temperature=0.05,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
            )
        
        # Add HuggingFace hub models if token is available
        if hf_api_token:
            try:
                # Import HuggingFaceEndpoint which handles raw responses better
                from langchain.llms import HuggingFaceEndpoint
                
                # Mixtral with proper handling for text generation
                self.add_model(
                    "mixtral-8x7b", 
                    HuggingFaceEndpoint(
                        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
                        huggingfacehub_api_token=hf_api_token,
                        task="text-generation",
                        model_kwargs={
                            "temperature": 0.05,
                            "max_new_tokens": 512,
                            "return_full_text": False
                        }
                    )
                )
                
                # FLAN-T5 with proper handling
                self.add_model(
                    "flan-t5-xl", 
                    HuggingFaceEndpoint(
                        endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-xl",
                        huggingfacehub_api_token=hf_api_token,
                        task="text2text-generation",
                        model_kwargs={
                            "temperature": 0.05,
                            "max_new_tokens": 512
                        }
                    )
                )
            except Exception as e:
                print(f"Could not load HuggingFace models: {e}")
                print("Falling back to basic HuggingFaceHub implementation...")
                
                # Fallback to standard implementation
                # Mixtral
                self.add_model(
                    "mixtral-8x7b", 
                    HuggingFaceHub(
                        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={"temperature": 0.05},
                        huggingfacehub_api_token=hf_api_token
                    )
                )
                
                # FLAN-T5
                self.add_model(
                    "flan-t5-xl", 
                    HuggingFaceHub(
                        repo_id="google/flan-t5-xl",
                        model_kwargs={"temperature": 0.05},
                        huggingfacehub_api_token=hf_api_token
                    )
                )

        # Try to add a local HF model if available
        try:
            model_name = "google/flan-t5-base"  # A smaller model that might run locally
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512
            )
            
            local_llm = HuggingFacePipeline(pipeline=pipe)
            self.add_model("local-flan-t5", local_llm)
        except Exception as e:
            print(f"Could not load local model: {e}")
            
    def add_model(self, model_name, model_instance):
        """Add a model to the comparison set.
        
        Args:
            model_name: Unique identifier for the model
            model_instance: LangChain compatible model instance
        """
        self.models[model_name] = model_instance
        print(f"Added model: {model_name}")
        
    def remove_model(self, model_name):
        """Remove a model from the comparison set."""
        if model_name in self.models:
            del self.models[model_name]
            print(f"Removed model: {model_name}")
        else:
            print(f"Model {model_name} not found")
            
    def list_models(self):
        """List all available models for comparison."""
        return list(self.models.keys())
        
    def sql_copilot(self, user_question, model_name, max_attempts=3):
        """Run the SQL copilot with a specific model.
        
        Args:
            user_question: The user's natural language query
            model_name: Name of the model to use
            max_attempts: Maximum number of attempts to debug SQL
            
        Returns:
            dict: Results including SQL, execution time, and answer
        """
        if model_name not in self.models:
            return {
                "status": "error",
                "message": f"Model {model_name} not found"
            }
            
        language_model = self.models[model_name]
        start_time = time.time()
        
        print(f"\n--- Using model: {model_name} ---")
        print("\nIdentifying most likely schemas...")
        db_documents = similar_doc_search(question=user_question, vector_database=vectordb, top_k=3)

        top_schemas = identify_schemas(db_documents)
        print(top_schemas)
        result = None
        
        for schema in list(top_schemas):
            print(f"\nConnecting to {schema} schema...")
            db = connect_db(db_path=db_filepath, target_schema=schema)
            print("...Connected to database.")
            print(db.get_usable_table_names())

            tables_sorted = prioritize_tables(documents=db_documents, target_schema=schema, sql_database=db)
            tables_info = get_table_info(tables=tables_sorted, database=db)
            sql_dialect = get_sql_dialect(database=db)

            print("\nCalling to language model...")
            try:
                sql_statement = llm_create_sql(
                    sql_dialect=sql_dialect, 
                    table_info=tables_info, 
                    question=user_question, 
                    lang_model=language_model
                )
                print(f"\nTry this SQL statement: {sql_statement}")
            except ValueError as err_msg:
                print(f"\n{str(err_msg)}")
                print('\nMoving on to try the next schema...')
                result = 'FAIL'
                continue

            print("\nValidating SQL...")
            try:
                validated_sql = llm_check_sql(
                    sql_query=sql_statement, 
                    sql_dialect=sql_dialect, 
                    lang_model=language_model
                )
                print("...SQL validated.")
            except Exception as e:
                print(f"SQL validation failed: {e}")
                validated_sql = sql_statement  # Fallback to original statement

            attempt = 1
            query_to_run = validated_sql
            print("\nRunning query on database...")
            while attempt <= max_attempts:
                print(f"Attempting query: {query_to_run}")
                query_result = run_sql(database=db, sql_query=query_to_run)

                if query_result[:5] == 'Error':
                    if attempt >= max_attempts:
                        result = 'FAIL'
                        output = {
                            "status": "error",
                            "message": f"Unable to execute the SQL query after {max_attempts} attempts.",
                            "final_query": query_to_run,
                            "error": query_result
                        }
                        break

                    print("\nThat query returned an error. Working on debugging...")
                    try:
                        query_to_run = llm_debug_error(
                            sql_query=query_to_run, 
                            error_message=query_result, 
                            lang_model=language_model
                        )
                    except Exception as e:
                        print(f"Error during debugging: {e}")
                        result = 'FAIL'
                        output = {
                            "status": "error",
                            "message": f"Error during SQL debugging: {str(e)}",
                            "final_query": query_to_run
                        }
                        break
                        
                    attempt += 1
                    time.sleep(1)
                    print("\nTrying again...")

                elif query_result == '[]':
                    if attempt >= max_attempts:
                        result = 'FAIL'
                        output = {
                            "status": "empty_result",
                            "message": f"Query returned blank results after {max_attempts} attempts.",
                            "final_query": query_to_run
                        }
                        break

                    print("\nThat query returned no results. Working on debugging...")
                    try:
                        query_to_run = llm_debug_empty(
                            sql_query=query_to_run, 
                            quesiton=user_question, 
                            lang_model=language_model
                        )
                    except Exception as e:
                        print(f"Error during empty result debugging: {e}")
                        result = 'FAIL'
                        output = {
                            "status": "error",
                            "message": f"Error during empty result debugging: {str(e)}",
                            "final_query": query_to_run
                        }
                        break
                        
                    attempt += 1
                    time.sleep(1)
                    print("\nTrying again...")

                else:
                    result = 'SUCCESS'
                    try:
                        llm_answer = llm_analyze(
                            query_result=query_result, 
                            question=user_question, 
                            lang_model=language_model
                        )
                    except Exception as e:
                        llm_answer = f"Error analyzing results: {str(e)}"

                    output = {
                        "status": "success",
                        "input_question": user_question,
                        "original_sql": sql_statement,
                        "final_sql": query_to_run,
                        "sql_output": query_result,
                        "answer": llm_answer,
                        "schema": schema,
                        "tables_used": tables_sorted
                    }
                    break

            if result == 'SUCCESS':
                break

        if result != 'SUCCESS':
            if 'output' not in locals():
                output = {
                    "status": "error",
                    "message": "Failed to find relevant schema or generate valid SQL"
                }
                
        # Add timing information
        execution_time = time.time() - start_time
        output["execution_time"] = execution_time
        output["model"] = model_name
        
        return output
        
    def run_comparison(self, user_question, models_to_use=None, max_attempts=3):
        """Run a comparison of the SQL copilot across multiple models.
        
        Args:
            user_question: The user's natural language query
            models_to_use: List of model names to use (defaults to all)
            max_attempts: Maximum SQL debugging attempts per model
            
        Returns:
            dict: Comparison results for all models
        """
        if not models_to_use:
            models_to_use = self.list_models()
        
        results = {}
        for model_name in models_to_use:
            print(f"\n{'='*50}")
            print(f"Running query with model: {model_name}")
            print(f"{'='*50}")
            
            try:
                model_result = self.sql_copilot(user_question, model_name, max_attempts)
                results[model_name] = model_result
            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "message": f"Exception occurred: {str(e)}",
                    "model": model_name
                }
                
        # Save the comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "question": user_question,
                "timestamp": timestamp,
                "results": results
            }, f, indent=2)
            
        print(f"\nResults saved to {filename}")
        return results
        
    def compare_results(self, results):
        """Generate a summary comparison of results from different models.
        
        Args:
            results: Dict with model results from run_comparison
            
        Returns:
            pd.DataFrame: Comparison summary
        """
        summary = []
        
        for model_name, result in results.items():
            row = {
                "model": model_name,
                "status": result.get("status", "unknown"),
                "execution_time": round(result.get("execution_time", 0), 2),
                "sql_generated": "original_sql" in result,
                "sql_executed": result.get("status") == "success",
                "answer_generated": "answer" in result and result["answer"] not in (None, ""),
            }
            summary.append(row)
            
        df = pd.DataFrame(summary)
        return df
        
    def analyze_previous_comparisons(self, n=5):
        """Analyze the most recent comparison files.
        
        Args:
            n: Number of most recent comparison files to analyze
            
        Returns:
            pd.DataFrame: Aggregated stats across comparisons
        """
        files = sorted(self.results_dir.glob("comparison_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        files = files[:n]  # Take n most recent
        
        if not files:
            return "No comparison files found"
            
        all_stats = []
        
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                
            question = data["question"]
            timestamp = data["timestamp"]
            
            for model_name, result in data["results"].items():
                row = {
                    "question": question,
                    "timestamp": timestamp,
                    "model": model_name,
                    "status": result.get("status", "unknown"),
                    "execution_time": round(result.get("execution_time", 0), 2),
                    "success": result.get("status") == "success"
                }
                all_stats.append(row)
                
        df = pd.DataFrame(all_stats)
        return df
        
def main():
    """Interactive testing of the model comparison system."""
    comparator = ModelComparison()
    
    print("\nAvailable models for comparison:")
    for model in comparator.list_models():
        print(f"- {model}")
        
    question = input("\nWhat would you like to know from your data?: ")
    
    print("\nRunning comparison across all models...")
    results = comparator.run_comparison(question)
    
    print("\nComparison Summary:")
    summary_df = comparator.compare_results(results)
    print(summary_df)
    
    # Print the model answers
    print("\nModel Answers:")
    for model_name, result in results.items():
        print(f"\n--- {model_name} ---")
        if result.get("status") == "success":
            print(f"Answer: {result.get('answer', 'No answer generated')}")
        else:
            print(f"Status: {result.get('status')} - {result.get('message', '')}")

if __name__ == '__main__':
    main()
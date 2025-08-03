import os
from dotenv import load_dotenv
import time
import argparse
from pathlib import Path

from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.llms import VertexAI, HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Import our custom HuggingFace wrapper
from custom_hf_wrapper import create_hf_model

from sql_functions import (
    similar_doc_search, identify_schemas, connect_db, prioritize_tables,
    get_table_info, get_sql_dialect, llm_create_sql, llm_check_sql,
    run_sql, llm_debug_error, llm_debug_empty, llm_analyze
)

# Import our comparison module
from model_comparison import ModelComparison

# Setup embeddings using HuggingFace and the directory location
embeddings = HuggingFaceEmbeddings()
persist_dir = '../data/processed/chromadb'
db_filepath = '../data/raw/spider/database/'

# Load from disk
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Get API key
load_dotenv()
hf_api_token = os.getenv('hf_token')

def get_language_model(model_name="gpt-3.5-turbo"):
    """Get a language model by name.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        A LangChain compatible model
    """
    if model_name == "gpt-3.5-turbo":
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.05,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name == "gpt-4":
        return ChatOpenAI(
            model_name="gpt-4",
            temperature=0.05,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name == "mixtral-8x7b":
        # Use our custom wrapper for Mixtral
        print("Using custom HuggingFace wrapper for Mixtral-8x7B")
        return create_hf_model("mixtral")
    elif model_name == "flan-t5-xl":
        # Use our custom wrapper for FLAN-T5
        print("Using custom HuggingFace wrapper for FLAN-T5-XL")
        return create_hf_model("flan-t5")
    elif model_name == "llama2-70b":
        # Use our custom wrapper for Llama 2
        print("Using custom HuggingFace wrapper for Llama-2-70B")
        return create_hf_model("llama2")
    elif model_name == "gemma":
        # Use our custom wrapper for Gemma
        print("Using custom HuggingFace wrapper for Gemma")
        return create_hf_model("gemma")
    elif model_name == "local-flan-t5":
        # Load a local HF model if available
        model_name = "google/flan-t5-xl"  # A smaller model that might run locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    else:
        # Default to GPT-3.5 if unknown model
        print(f"Unknown model: {model_name}, defaulting to gpt-3.5-turbo")
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.05,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def sql_copilot(user_question: str, model_name="gpt-3.5-turbo", max_attempts=3):
    """Run SQL Copilot with specified model.
    
    Args:
        user_question: User's natural language query
        model_name: Name of the LLM to use
        max_attempts: Maximum SQL debugging attempts
        
    Returns:
        str: Formatted output with SQL and results
    """
    language_model = get_language_model(model_name)
    print(f"\nUsing model: {model_name}")
    
    print("\nIdentifying most likely schemas...")
    db_documents = similar_doc_search(question=user_question, vector_database=vectordb, top_k=3)

    top_schemas = identify_schemas(db_documents)
    print(top_schemas)
    result = None

    for schema in list(top_schemas):
        print("\nConnecting to " + schema + " schema...")
        db = connect_db(db_path=db_filepath, target_schema=schema)
        print("...Connected to database.")
        print(db.get_usable_table_names())

        tables_sorted = prioritize_tables(documents=db_documents, target_schema=schema, sql_database=db)
        tables_info = get_table_info(tables=tables_sorted, database=db)
        sql_dialect = get_sql_dialect(database=db)

        print("\nCalling to language model...")
        try:
            sql_statement = llm_create_sql(sql_dialect=sql_dialect, table_info=tables_info, question=user_question, lang_model=language_model)
            print("\nTry this SQL statement: " + sql_statement)
        except ValueError as err_msg:
            print("\n" + str(err_msg))
            print('\nMoving on to try the next schema...')
            result = 'FAIL'
            continue

        print("\nValidating SQL...")
        validated_sql = llm_check_sql(sql_query=sql_statement, sql_dialect=sql_dialect, lang_model=language_model)
        print("...SQL validated.")

        attempt = 1
        query_to_run = validated_sql
        print("\nRunning query on database...")
        while attempt <= max_attempts:
            print("Attempting query:", query_to_run)
            query_result = run_sql(database=db, sql_query=query_to_run)

            if query_result[:5] == 'Error':
                if attempt >= max_attempts:
                    result = 'FAIL'
                    output = f"Unable to execute the SQL query after {max_attempts} attempts."
                    break

                print("\nThat query returned an error. Working on debugging...")
                query_to_run = llm_debug_error(sql_query=query_to_run, error_message=query_result, lang_model=language_model)
                attempt += 1
                time.sleep(1)
                print("\nTrying again...")

            elif query_result == '[]':
                if attempt >= max_attempts:
                    result = 'FAIL'
                    output = f"['']\nQuery returned blank results after {max_attempts} attempts."
                    break

                print("\nThat query returned no results. Working on debugging...")
                query_to_run = llm_debug_empty(sql_query=query_to_run, quesiton=user_question, lang_model=language_model)
                attempt += 1
                time.sleep(1)
                print("\nTrying again...")

            else:
                result = 'SUCCESS'
                llm_answer = llm_analyze(query_result=query_result, question=user_question, lang_model=language_model)

                output = f"""
                    Input Question: {user_question}
                    SQL Query: {query_to_run}
                    SQL Output: {query_result}
                    Answer: {llm_answer}"""
                break

        if result == 'SUCCESS':
            break

    if result == 'SUCCESS':
        print("\nHere is what I found:")
        print(output)
        return output
    else:
        print("Sorry, I was not able to find the answer to your question.")
        return "Sorry, I was not able to find the answer to your question."

def main():
    parser = argparse.ArgumentParser(description='SQL Copilot')
    parser.add_argument('--question', '-q', type=str, help='Question to ask')
    parser.add_argument('--model', '-m', type=str, default='gpt-3.5-turbo', 
                        help='Model to use (gpt-3.5-turbo, gpt-4, mixtral-8x7b, flan-t5-xl, local-flan-t5)')
    parser.add_argument('--compare', '-c', action='store_true', 
                        help='Run comparison across multiple models')
    parser.add_argument('--models', type=str, 
                        help='Comma-separated list of models to compare')
    
    args = parser.parse_args()
    
    if args.question:
        question = args.question
    else:
        question = input("What would you like to know from your data?: ")
    
    if args.compare:
        # Run comparison across models
        comparator = ModelComparison()
        
        if args.models:
            # Use specific models
            models_to_use = [m.strip() for m in args.models.split(',')]
        else:
            # Use all available models
            models_to_use = comparator.list_models()
            
        print(f"\nRunning comparison across models: {', '.join(models_to_use)}")
        results = comparator.run_comparison(question, models_to_use)
        
        print("\nComparison Summary:")
        summary_df = comparator.compare_results(results)
        print(summary_df)
        
        # Print model answers
        print("\nModel Answers:")
        for model_name, result in results.items():
            print(f"\n=== {model_name} ===")
            if result.get("status") == "success":
                print(f"Answer: {result.get('answer', 'No answer generated')}")
                print(f"SQL: {result.get('final_sql', 'No SQL generated')}")
            else:
                print(f"Status: {result.get('status')} - {result.get('message', '')}")
    else:
        # Run with single model
        sql_copilot(user_question=question, model_name=args.model)

if __name__ == '__main__':
    main()
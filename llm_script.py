# llm_script.py
import pandas as pd
import openai
import os
import sys
import argparse
import math
from dotenv import load_dotenv

def create_chunks(df, chunk_size=200):
    """Splits a DataFrame into smaller chunks."""
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        yield df.iloc[i*chunk_size:(i+1)*chunk_size]

def analyze_data(file_path):
    """
    Reads data from a file, performs LLM analysis (with chunking for large files),
    and prints the result.
    """
    try:
        load_dotenv()

        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(file_path)
        
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=300.0, # Increased timeout for longer processing
        )

        # Use chunking logic for large files
        if len(df) > 200:
            chunk_summaries = []
            
            # Map step: Analyze each chunk
            num_chunks = math.ceil(len(df) / 200)
            for i, chunk_df in enumerate(create_chunks(df, chunk_size=200)):
                chunk_data = chunk_df.to_string()
                
                # Progress indicator to stderr
                print(f"Analyzing chunk {i+1}/{num_chunks}...", file=sys.stderr)

                map_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst. You will be given a small chunk of a larger CSV file. Your task is to briefly summarize the key information, patterns, or anomalies in this specific chunk in Korean. The summary should be concise and clear, as it will be used in a later step to create a final, comprehensive report. Focus only on the provided data chunk."},
                        {"role": "user", "content": f"Please analyze the following data chunk and provide a summary:\n\n{chunk_data}"}
                    ],
                    temperature=0.2,
                )
                summary = map_response.choices[0].message.content
                chunk_summaries.append(summary)

            # Reduce step: Combine summaries for a final analysis
            print("Synthesizing final report...", file=sys.stderr)
            combined_summaries = "\n\n---\n\n".join(chunk_summaries)
            
            reduce_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior data analyst. You have been provided with a series of summaries from different chunks of a large CSV file. Your task is to synthesize these individual summaries into a single, cohesive, and well-structured final analysis in Korean. Identify overall trends, key insights, and any major anomalies or patterns found across the entire dataset. Provide a clear, actionable summary for a business user."},
                    {"role": "user", "content": f"Here are the summaries from the data chunks. Please create a final, comprehensive report based on them:\n\n{combined_summaries}"}
                ],
                temperature=0.5,
            )
            
            final_result = reduce_response.choices[0].message.content
            print(final_result) # Print final result to stdout
        
        else: # Process small files directly
            data_for_analysis = df.to_string()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Analyze the following data and provide a detailed summary and insights in Korean."},
                    {"role": "user", "content": f"Analyze the following data and provide insights:\n\n{data_for_analysis}"}
                ]
            )
            result = response.choices[0].message.content
            print(result) # Print result to stdout
        
    except Exception as e:
        print(f"Error during LLM analysis: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing file {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Analysis Script")
    parser.add_argument("--file", required=True, help="Path to the data file for analysis")
    args = parser.parse_args()
    
    analyze_data(args.file)

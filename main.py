import os
import openai
import pandas as pd
from typing import Optional, List, Dict
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file if it exists
load_dotenv()

def load_dev_data(csv_file: str = "json_list.csv", phase_filter: str = None) -> pd.DataFrame:
    """
    Load developer data from CSV file with optional phase filtering.
    
    Args:
        csv_file (str): Path to the CSV file
        phase_filter (str, optional): Filter by specific phase (e.g., "2. Phase 2")
        
    Returns:
        pd.DataFrame: DataFrame containing developer data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        Exception: For other CSV parsing errors
    """
    try:
        # Load CSV with proper handling of multi-line text in resume_plain column
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Clean up column names (remove any whitespace)
        df.columns = df.columns.str.strip()
        
        print(f"Successfully loaded {len(df)} total developer records")
        print(f"Columns: {list(df.columns)}")
        
        # Apply phase filter if specified and column exists
        if phase_filter:
            if 'phase' in df.columns:
                original_count = len(df)
                df = df[df['phase'] == phase_filter]
                print(f"Filtered to {len(df)} developers in '{phase_filter}'")
                print(f"Removed {original_count - len(df)} developers from other phases")
            else:
                print("'phase' column not found; skipping phase filter.")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please make sure the file exists in the current directory.")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {e}")

def get_developer_by_id(df: pd.DataFrame, developer_id: int) -> Dict:
    """
    Get developer data by ID.
    
    Args:
        df (pd.DataFrame): DataFrame containing developer data
        developer_id (int): Developer ID to search for
        
    Returns:
        Dict: Developer data as dictionary
        
    Raises:
        ValueError: If developer ID not found
    """
    developer = df[df['developer_id'] == developer_id]
    
    if developer.empty:
        raise ValueError(f"Developer with ID {developer_id} not found")
    
    return developer.iloc[0].to_dict()

def evaluate_resume_with_openai(developer_data: Dict, prompt_template: str) -> str:
    """
    Evaluate a developer's resume using OpenAI API with the evaluation prompt.
    
    Args:
        developer_data (Dict): Developer data containing resume_plain, country, and other fields
        prompt_template (str): The evaluation prompt template from prompt.txt
        
    Returns:
        str: OpenAI API response with evaluation results
    """
    # Replace the placeholders in the prompt with actual data
    evaluation_prompt = prompt_template.replace("{dev_resume}", developer_data.get('resume_plain', ''))
    evaluation_prompt = evaluation_prompt.replace("{country}", developer_data.get('country_name', ''))
    
    # Call OpenAI API with the evaluation prompt
    return call_openai_api(evaluation_prompt)

def load_evaluation_prompt(prompt_file: str = "prompt.txt") -> str:
    """
    Load the evaluation prompt template from file.
    
    Args:
        prompt_file (str): Path to the prompt template file
        
    Returns:
        str: The evaluation prompt template
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found. Please make sure the file exists in the current directory.")

def evaluate_all_developers(df: pd.DataFrame, prompt_template: str, max_developers: int = None, output_file: str = "evaluation_results.json", max_workers: int = 5, merge_with_existing: bool = False) -> List[Dict]:
    """
    Evaluate all developers in the DataFrame using OpenAI API with parallel processing and incremental saving.
    
    Args:
        df (pd.DataFrame): DataFrame containing developer data
        prompt_template (str): The evaluation prompt template
        max_developers (int): Maximum number of developers to evaluate (for cost control)
        output_file (str): File to save incremental results
        max_workers (int): Number of parallel workers (default: 5)
        merge_with_existing (bool): If True, merge new results with existing results from output_file
        
    Returns:
        List[Dict]: List of evaluation results with developer info and OpenAI response
    """
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    
    # Load existing results if merging
    existing_results = []
    if merge_with_existing:
        existing_results = load_existing_results(output_file)
        existing_ids = {r.get('developer_id') for r in existing_results if 'developer_id' in r}
        print(f"📋 Merging with {len(existing_results)} existing results")
    
    results = []
    results_lock = Lock()
    
    # Limit the number of developers to evaluate (if specified)
    if max_developers is not None:
        developers_to_evaluate = df.head(max_developers)
    else:
        developers_to_evaluate = df  # Process all rows
    total_developers = len(developers_to_evaluate)
    
    # Progress tracking
    completed_count = 0
    progress_lock = Lock()
    
    def evaluate_single_developer(developer_data):
        """Evaluate a single developer's resume."""
        try:
            # Replace placeholders in the prompt template
            evaluation_prompt = prompt_template.replace("{dev_resume}", developer_data.get('resume_plain', ''))
            evaluation_prompt = evaluation_prompt.replace("{country}", developer_data.get('country_name', ''))
            
            # Evaluate the resume
            evaluation_result = call_openai_api(evaluation_prompt)
            
            # Create result
            result = {
                'developer_id': developer_data.get('developer_id'),
                'evaluation': evaluation_result
            }
            
            return result
            
        except Exception as e:
            # Return error result
            return {
                'developer_id': developer_data.get('developer_id'),
                'evaluation': f"ERROR: {str(e)}"
            }
    
    def update_progress():
        """Update progress bar."""
        nonlocal completed_count
        with progress_lock:
            completed_count += 1
            progress = completed_count / total_developers
            filled_length = int(20 * progress)
            bar = '█' * filled_length + '-' * (20 - filled_length)
            print(f"\rProgress: [{bar}] {progress*100:.0f}% ({completed_count}/{total_developers})", end="", flush=True)
    
    # Progress bar setup
    print(f"Evaluating {total_developers} developers with {max_workers} parallel workers...")
    print("Progress: ", end="", flush=True)
    
    # Convert DataFrame to list of dictionaries
    developer_list = developers_to_evaluate.to_dict('records')
    
    # Process developers in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_developer = {
            executor.submit(evaluate_single_developer, dev): dev 
            for dev in developer_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_developer):
            try:
                result = future.result()
                
                # Add to results thread-safely
                with results_lock:
                    results.append(result)
                    
                    # Incrementally save results (merge with existing if needed)
                    if merge_with_existing:
                        # Create a dictionary of existing results by developer_id for quick lookup
                        existing_dict = {r.get('developer_id'): r for r in existing_results if 'developer_id' in r}
                        # Update existing results with new ones
                        for new_result in results:
                            dev_id = new_result.get('developer_id')
                            if dev_id is not None:
                                existing_dict[dev_id] = new_result
                        # Combine: existing (not updated) + new results
                        all_results = list(existing_dict.values())
                    else:
                        all_results = results
                    
                    # Incrementally save results
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                
                # Update progress
                update_progress()
                
            except Exception as e:
                # Handle any remaining errors
                update_progress()
                continue
    
    print()  # New line after progress bar
    
    # Return merged results if needed
    if merge_with_existing:
        existing_dict = {r.get('developer_id'): r for r in existing_results if 'developer_id' in r}
        for new_result in results:
            dev_id = new_result.get('developer_id')
            if dev_id is not None:
                existing_dict[dev_id] = new_result
        return list(existing_dict.values())
    
    return results

def save_evaluation_results(results: List[Dict], output_file: str = "evaluation_results.json"):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results (List[Dict]): List of evaluation results
        output_file (str): Output file path
    """
    import json
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Evaluation results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def parse_evaluation_response(response: str) -> Dict:
    """
    Parse the JSON response from OpenAI evaluation.
    
    Args:
        response (str): The response string from OpenAI containing JSON
        
    Returns:
        Dict: Parsed evaluation scores
        
    Raises:
        ValueError: If JSON parsing fails
    """
    import json
    import re
    
    try:
        # Extract JSON from the response (remove markdown code blocks if present)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without markdown
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        # Parse the JSON
        evaluation_data = json.loads(json_str)
        return evaluation_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")

def create_evaluation_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Create a DataFrame from evaluation results with developer_id and parsed scores
    matching the current evaluation prompt (pedigree flags: top companies and universities).
    
    Args:
        results (List[Dict]): List of evaluation results from evaluate_all_developers
        
    Returns:
        pd.DataFrame: DataFrame with developer_id and evaluation scores
    """
    # Define all expected fields from the prompt
    expected_fields1 = [
        'Primary_Category',
        'Subcategory',
        'Role_or_Specialty'
    ]

    expected_fields = [
        'top_companies_faang',
        'top_company_50',
        'top_company_50_names',
        'top_university_us',
        'top_university_us_names',
        'top_university_whole_list',
        'top_university_whole_list_names',
        'top_company_50_2yoe'
    ]
    
    evaluation_rows = []
    
    for result in results:
        try:
            # Parse the evaluation response
            evaluation_scores = parse_evaluation_response(result['evaluation'])
            
            # Create row with developer info and scores
            row = {
                'developer_id': result.get('developer_id')
            }
            
            # Add all evaluation scores
            row.update(evaluation_scores)
            
            evaluation_rows.append(row)
            
        except Exception as e:
            print(f"Error parsing evaluation for developer {result['developer_id']}: {e}")
            # Add row with error indicator for all expected fields
            error_row = {
                'developer_id': result.get('developer_id')
            }
            # Set all fields to ERROR
            for field in expected_fields:
                error_row[field] = 'ERROR'
            evaluation_rows.append(error_row)
    
    return pd.DataFrame(evaluation_rows)

def show_phase_distribution(csv_file: str = "json_list.csv"):
    """
    Show the distribution of developers across different phases.
    
    Args:
        csv_file (str): Path to the CSV file
    """
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        df.columns = df.columns.str.strip()
        
        total_developers = len(df)
        if 'phase' in df.columns:
            phase_counts = df['phase'].value_counts()
            print(f"\n📊 Phase Distribution (Total: {total_developers} developers):")
            print("=" * 50)
            for phase, count in phase_counts.items():
                percentage = (count / total_developers) * 100
                print(f"   {phase}: {count} developers ({percentage:.1f}%)")
            return phase_counts
        else:
            print(f"\n📊 Dataset Overview: {total_developers} developers")
            print("'phase' column not found; skipping phase breakdown.")
            return None
        
    except Exception as e:
        print(f"Error showing phase distribution: {e}")
        return None

def save_evaluation_dataframe(df: pd.DataFrame, output_file: str = "evaluation_scores.csv"):
    """
    Save evaluation DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame with evaluation scores
        output_file (str): Output CSV file path
    """
    try:
        df.to_csv(output_file, index=False)
        print(f"Evaluation scores saved to {output_file}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")

def estimate_processing_info(df: pd.DataFrame, max_developers: int = None) -> Dict:
    """
    Estimate processing time and cost for evaluation.
    
    Args:
        df (pd.DataFrame): DataFrame with developer data
        max_developers (int): Maximum number of developers to process
        
    Returns:
        Dict: Estimated processing information
    """
    total_rows = len(df)
    if max_developers is not None:
        rows_to_process = min(max_developers, total_rows)
    else:
        rows_to_process = total_rows
    
    # Rough estimates (adjust based on your experience)
    avg_time_per_request = 2  # seconds (including API call + processing, with parallel processing)
    estimated_cost_per_request = 0.01  # USD (rough estimate for GPT-4o)
    
    estimated_time_minutes = (rows_to_process * avg_time_per_request) / 60
    estimated_cost = rows_to_process * estimated_cost_per_request
    
    return {
        'total_rows': total_rows,
        'rows_to_process': rows_to_process,
        'estimated_time_minutes': estimated_time_minutes,
        'estimated_cost_usd': estimated_cost
    }

def confirm_processing(df: pd.DataFrame, max_developers: int = None) -> bool:
    """
    Ask for confirmation before processing large datasets.
    
    Args:
        df (pd.DataFrame): DataFrame with developer data
        max_developers (int): Maximum number of developers to process
        
    Returns:
        bool: True if user confirms, False otherwise
    """
    info = estimate_processing_info(df, max_developers)
    
    print(f"\n📊 Processing Information:")
    print(f"   Total rows in CSV: {info['total_rows']}")
    print(f"   Rows to process: {info['rows_to_process']}")
    print(f"   Estimated time: {info['estimated_time_minutes']:.1f} minutes")
    print(f"   Estimated cost: ${info['estimated_cost_usd']:.2f} USD")
    
    if info['rows_to_process'] > 10:
        print(f"\n⚠️  Large dataset detected! This will take time and cost money.")
        print(f"   Consider testing with a smaller subset first.")
        
        response = input(f"\nContinue processing {info['rows_to_process']} developers? (y/N): ").strip().lower()
        return response in ['y', 'yes']
    
    return True

def load_existing_developer_ids(results_file: str = "evaluation_results.json") -> set:
    """
    Load developer_ids from existing evaluation results file.
    
    Args:
        results_file (str): Path to the evaluation results JSON file
        
    Returns:
        set: Set of developer_ids that have already been evaluated
    """
    import json
    
    existing_ids = set()
    
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                
            # Extract developer_ids from existing results
            for result in existing_results:
                if 'developer_id' in result:
                    existing_ids.add(result['developer_id'])
            
            print(f"📋 Loaded {len(existing_ids)} existing developer_ids from {results_file}")
        else:
            print(f"📋 No existing results file found at {results_file}")
            
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Could not parse {results_file}: {e}")
        print(f"   Will process all developers (no exclusions)")
    except Exception as e:
        print(f"⚠️  Warning: Error loading {results_file}: {e}")
        print(f"   Will process all developers (no exclusions)")
    
    return existing_ids

def load_existing_results(results_file: str = "evaluation_results.json") -> List[Dict]:
    """
    Load existing evaluation results from file.
    
    Args:
        results_file (str): Path to the evaluation results JSON file
        
    Returns:
        List[Dict]: List of existing evaluation results
    """
    import json
    
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            return existing_results
        else:
            return []
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing results from {results_file}: {e}")
        return []

def verify_prompt_replacement(developer_data: Dict, prompt_template: str) -> bool:
    """
    Verify that the prompt replacement is working correctly.
    
    Args:
        developer_data (Dict): Developer data
        prompt_template (str): Original prompt template
        
    Returns:
        bool: True if replacement works correctly
    """
    # Check if placeholders exist in template
    if "{dev_resume}" not in prompt_template:
        print("❌ ERROR: {dev_resume} placeholder not found in prompt template")
        return False
    
    if "{country}" not in prompt_template:
        print("❌ ERROR: {country} placeholder not found in prompt template")
        return False
    
    # Check if resume data exists
    if 'resume_plain' not in developer_data:
        print("❌ ERROR: resume_plain not found in developer data")
        return False
    
    # Perform replacement
    evaluation_prompt = prompt_template.replace("{dev_resume}", developer_data.get('resume_plain', ''))
    evaluation_prompt = evaluation_prompt.replace("{country}", developer_data.get('country', ''))
    
    # Check if replacement worked
    if "{dev_resume}" in evaluation_prompt:
        print("❌ ERROR: {dev_resume} placeholder still exists after replacement")
        return False
    
    if "{country}" in evaluation_prompt:
        print("❌ ERROR: {country} placeholder still exists after replacement")
        return False
    
    if developer_data.get('resume_plain', '') not in evaluation_prompt:
        print("❌ ERROR: Resume data not found in final prompt")
        return False
    
    print("✅ Prompt replacement verified successfully")
    return True

def call_openai_api(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Call OpenAI API with a simple prompt.
    
    Args:
        prompt (str): The prompt to send to OpenAI
        api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable
    
    Returns:
        str: The response from OpenAI API
        
    Raises:
        ValueError: If no API key is provided
        Exception: For other API errors
    """
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Either pass it as a parameter or set OPENAI_API_KEY environment variable.")
    
    # Set up the OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            #model="gpt-3.5-turbo",  # You can change this to other models like "gpt-4"
            model="gpt-4o",  # You can change this to other models like "gpt-4"
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  # Adjust as needed
            temperature=0.7   # Adjust creativity level (0.0 = deterministic, 1.0 = very creative)
        )
        
        # Extract and return the response
        return response.choices[0].message.content
        
    except openai.AuthenticationError:
        raise Exception("Invalid API key. Please check your OpenAI API key.")
    except openai.RateLimitError:
        raise Exception("Rate limit exceeded. Please try again later.")
    except openai.APIError as e:
        raise Exception(f"OpenAI API error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # CLI arguments
        parser = argparse.ArgumentParser(description="Evaluate developer resumes using OpenAI")
        parser.add_argument("--csv", type=str, default="json_list.csv", help="Path to input CSV (default: json_list.csv)")
        parser.add_argument("--limit", type=int, default=None, help="Limit number of developers to process (e.g., 1 for single row)")
        parser.add_argument("--developer_id", type=int, default=None, help="Process only the specified developer_id")
        parser.add_argument("--exclude-existing", action="store_true", help="Exclude developer_ids that are already in evaluation_results.json")
        parser.add_argument("--results-file", type=str, default="evaluation_results.json", help="Path to evaluation results file (default: evaluation_results.json)")
        args = parser.parse_args()

        # Load developers (dataset contains only developer_id and resume_plain)
        df = load_dev_data(csv_file=args.csv)

        # Optional: filter to a specific developer_id
        if args.developer_id is not None:
            df = df[df['developer_id'] == args.developer_id]
            print(f"\nProcessing developer_id={args.developer_id} from {args.csv}")
        else:
            print(f"\nProcessing {len(df)} developers from {args.csv}")
        
        # Optional: exclude existing developer_ids
        if args.exclude_existing:
            existing_ids = load_existing_developer_ids(args.results_file)
            if existing_ids:
                original_count = len(df)
                df = df[~df['developer_id'].isin(existing_ids)]
                excluded_count = original_count - len(df)
                print(f"🔍 Excluded {excluded_count} developers that are already in {args.results_file}")
                print(f"📊 Remaining developers to process: {len(df)}")
            else:
                print(f"📊 No existing results found; processing all {len(df)} developers")
        
        # Load the evaluation prompt template
        prompt_template = load_evaluation_prompt()
        
        # Example 2: Evaluate multiple developers and create DataFrame
        if len(df) > 0:
            # Show processing information and get confirmation
            if confirm_processing(df, max_developers=args.limit):
                # Evaluate all developers with parallel processing and incremental saving
                results = evaluate_all_developers(
                    df, 
                    prompt_template, 
                    max_developers=args.limit, 
                    output_file=args.results_file, 
                    max_workers=5,
                    merge_with_existing=args.exclude_existing
                )
                
                # Create DataFrame with parsed scores
                evaluation_df = create_evaluation_dataframe(results)
                
                # Save final DataFrame
                save_evaluation_dataframe(evaluation_df, "evaluation_scores.csv")
                
                # Show final summary
                print(f"\n✅ Completed evaluation of {len(results)} developers")
                print(f"📁 Results saved to: {args.results_file} and evaluation_scores.csv")
            else:
                print("❌ Processing cancelled by user")
        else:
            print("❌ No developers found in the dataset")
            
    except Exception as e:
        print(f"Error: {e}")


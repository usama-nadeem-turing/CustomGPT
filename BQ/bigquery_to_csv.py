import argparse
import os
from typing import Optional

import pandas as pd
from google.cloud import bigquery


def run_query_to_csv(
    sql_file: str,
    output_csv: str,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> None:
    """
    Run a BigQuery SQL query from a file and save the results to a CSV file.

    Args:
        sql_file: Path to the .sql file containing the query.
        output_csv: Path to the output CSV file.
        project_id: Optional GCP project id. If not provided, BigQuery client
                    will try to infer it from the environment.
        location: Optional BigQuery location (e.g. "US", "EU").
    """
    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    with open(sql_file, "r", encoding="utf-8") as f:
        query = f.read().strip()

    if not query:
        raise ValueError(f"SQL file {sql_file} is empty")

    client_kwargs = {}
    if project_id:
        client_kwargs["project"] = project_id
    client = bigquery.Client(**client_kwargs)

    job_config = bigquery.QueryJobConfig()
    job = client.query(query, job_config=job_config, location=location)

    print(f"Running BigQuery query from {sql_file}...")
    result = job.result()  # Waits for job to complete
    print(f"Query completed. {result.total_rows} rows returned.")

    # Convert to DataFrame and write to CSV
    df = result.to_dataframe()
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a BigQuery SQL query from a file and save results to CSV."
    )
    parser.add_argument(
        "--sql-file",
        type=str,
        default="bigquery_query.sql",
        help="Path to SQL file with the BigQuery query (default: bigquery_query.sql)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="bigquery_results.csv",
        help="Path to output CSV file (default: bigquery_results.csv)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="GCP project ID (optional, will use default project if omitted)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help='BigQuery location, e.g. "US" or "EU" (optional)',
    )

    args = parser.parse_args()

    # Note: Authentication should be handled via GOOGLE_APPLICATION_CREDENTIALS
    # env var or gcloud auth application-default login.
    # Example:
    #   set GOOGLE_APPLICATION_CREDENTIALS=path\to\service-account.json   (Windows PowerShell)

    run_query_to_csv(
        sql_file=args.sql_file,
        output_csv=args.output_csv,
        project_id=args.project_id,
        location=args.location,
    )


if __name__ == "__main__":
    main()



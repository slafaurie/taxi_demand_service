from datetime import date
import argparse

from src.etl.dwh import generate_connection, create_pickup_table
from src.etl.transform import batch_etl
from src.logger import get_logger

logger = get_logger('main_dwh')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DWH operations")
    
    # Create mutually exclusive groups for operations
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--create-tables', action='store_true',
                               help='Create database tables')
    operation_group.add_argument('--extract-data', action='store_true',
                               help='Run batch ETL process')
    
    # Date range arguments
    parser.add_argument("--from-date", type=date.fromisoformat,
                       help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to-date", type=date.fromisoformat,
                       help="End date in YYYY-MM-DD format")
    
    args = parser.parse_args()

    # Handle operations
    if args.create_tables:
        with generate_connection() as db:
            create_pickup_table(db)
        logger.info("Tables created successfully!")
    
    elif args.extract_data:
        if not (args.from_date and args.to_date):
            parser.error("--from-date and --to-date are required for batch ETL")
        batch_etl(args.from_date, args.to_date)
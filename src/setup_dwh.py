from src.etl.dwh import generate_connection, create_pickup_table
from src.logger import get_logger

if __name__=="__main__":
    logger = get_logger('dwh setup')
    with generate_connection() as db:
        create_pickup_table(db)
    logger.info("All Tables created successfully!")
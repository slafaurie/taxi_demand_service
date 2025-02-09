import polars as pl
from polars.testing import assert_frame_equal
import pytest
from pathlib import Path
from datetime import datetime,date
from src.repository.local_repo import LocalRepository
from src.etl.models import NYCPickupHourlySchema

@pytest.fixture
def temp_repo_path(tmp_path):
    pickup_path = tmp_path / "pickup_data"
    pickup_path.mkdir()
    return tmp_path

@pytest.fixture
def test_repo(temp_repo_path):
    test_repo = LocalRepository(temp_repo_path)
    test_repo.create_tables()
    return test_repo

def test_deduplicate_pickup_data(test_repo):
    # Create initial data
    initial_data = {
        "key": ["A_2023_01", "B_2023_01", "C_2023_01"],
        "pickup_datetime_hour": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0)
        ],
        "num_pickups": [10, 20, 30],
        "pickup_location_id": [1, 2, 3]
    }
    
    # Create new data with some overlapping keys but different values
    new_data = {
        "key": ["B_2023_01", "C_2023_01", "D_2023_01"],
        "pickup_datetime_hour": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0)
        ],
        "num_pickups": [25, 35, 40],  # Changed values for B and C
        "pickup_location_id": [2, 3, 4]
    }
    
    # Expected result after deduplication
    expected_data = {
        "key": ["A_2023_01", "B_2023_01", "C_2023_01", "D_2023_01"],
        "pickup_datetime_hour": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 0)
        ],
        "num_pickups": [10, 25, 35, 40],
        "pickup_location_id": [1, 2, 3, 4]
    }
    
    # Convert to DataFrames and enforce schema
    initial_df = NYCPickupHourlySchema.enforce_schema(pl.DataFrame(initial_data))
    new_df = NYCPickupHourlySchema.enforce_schema(pl.DataFrame(new_data))
    expected_df = NYCPickupHourlySchema.enforce_schema(pl.DataFrame(expected_data))
    
    # Execute deduplication
    result_df = test_repo._deduplicate_pickup_data(new_df, initial_df)  # noqa
    
    # Assert the result matches expected DataFrame
    assert_frame_equal(result_df, expected_df)


def test_upsert_pickup_data_empty_initial(test_repo):
    # Test upserting when no initial data exists
    new_data = {
        "key": ["A_2023_01"],
        "pickup_datetime_hour": [datetime(2023, 1, 1, 10, 0, 0)],
        "num_pickups": [10],
        "pickup_location_id": [1]
    }
    
    new_df = NYCPickupHourlySchema.enforce_schema(pl.DataFrame(new_data))
        
    # Perform upsert
    test_repo.upsert_pickup_data(new_df)
    
    # Read and verify result
    result_df = pl.read_parquet(test_repo._pickup_table)  # noqa
    assert_frame_equal(result_df, new_df) 

def test_fetch_pickup_data(test_repo):
    # Create test data
    test_data = {
        "key": ["A_2023_01", "B_2023_01", "C_2023_01", "D_2023_01"],
        "pickup_datetime_hour": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 11, 0, 0),
            datetime(2023, 1, 2, 10, 0, 0),
            datetime(2023, 1, 2, 11, 0, 0)
        ],
        "num_pickups": [10, 20, 30, 40],
        "pickup_location_id": [1, 2, 1, 2]
    }
    
    # Save test data
    input_df = NYCPickupHourlySchema.enforce_schema(pl.DataFrame(test_data))
    test_repo.upsert_pickup_data(input_df)
    
    # Test 1: Fetch data for specific location
    expected_df = pl.DataFrame({
        "key": ["A_2023_01", "C_2023_01"],
        "pickup_datetime_hour": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 2, 10, 0, 0)
        ],
        "num_pickups": [10, 30],
        "pickup_location_id": [1, 1]
    })
    expected_df = NYCPickupHourlySchema.enforce_schema(expected_df)
    
    result_df = test_repo.fetch_pickup_data(
        from_date = datetime(2000,1,1),
        to_date = datetime(2099,1,1),
        pickup_locations=[1]
    )
    assert_frame_equal(result_df, expected_df)
    
    # Test 2: Fetch data for date range
    expected_df = pl.DataFrame({
        "key": ["A_2023_01", "B_2023_01"],
        "pickup_datetime_hour": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 11, 0, 0)
        ],
        "num_pickups": [10, 20],
        "pickup_location_id": [1, 2]
    })
    expected_df = NYCPickupHourlySchema.enforce_schema(expected_df)
    
    result_df = test_repo.fetch_pickup_data(
        from_date=datetime(2023, 1, 1),
        to_date=datetime(2023, 1, 1, 23, 59, 59)
    )
    assert_frame_equal(result_df, expected_df)
    
    # Test 3: Fetch data with both location and date filters
    expected_df = pl.DataFrame({
        "key": ["B_2023_01"],
        "pickup_datetime_hour": [datetime(2023, 1, 1, 11, 0, 0)],
        "num_pickups": [20],
        "pickup_location_id": [2]
    })
    expected_df = NYCPickupHourlySchema.enforce_schema(expected_df)
    
    result_df = test_repo.fetch_pickup_data(
        pickup_locations=[2],
        from_date=datetime(2023, 1, 1),
        to_date=datetime(2023, 1, 1, 23, 59, 59)
    )
    assert_frame_equal(result_df, expected_df)


def test_fetch_pickup_data_invalid_date_range(test_repo):
    # Test with end_date before start_date
    with pytest.raises(ValueError):
        test_repo.fetch_pickup_data(
            from_date=datetime(2023, 1, 2),
            to_date=datetime(2023, 1, 1)
        )
    

import polars as pl
import pytest
from src.model.pipeline import get_time_lags

def test_get_time_lags_get_correct_lag_value():
    # Create sample data
    data = {
        'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
               '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'unique_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'y': [10, 20, 30, 40, 15, 25, 35, 45]
    }
    
    df = pl.DataFrame(data)
    n_lags = [1, 2]
    
    # Execute function
    result_df = get_time_lags(df, lags=n_lags)
    
    # Check specific lag values for series A
    a_series = result_df.filter(pl.col('unique_id') == 'A')
    assert a_series.get_column('y__1__lag').to_list()[-1] == 30  # Last value should be previous day's value
    assert a_series.get_column('y__2__lag').to_list()[-1] == 20  # Last value should be two days ago value

@pytest.mark.parametrize(
    "data_size, unique_ids, n_lags",
    [
        # Test case 1: 2 IDs, 5 days each, 2 lags
        (5, ['A', 'B'], [1, 2]),
        # Test case 2: 3 IDs, 7 days each, 3 lags
        (7, ['A', 'B', 'C'], [1, 2, 3]),
        # Test case 3: 4 IDs, 10 days each, 5 lags
        (10, ['A', 'B', 'C', 'D'], [1, 2, 3, 4, 5]),
        # Test case 4: 1 ID, 15 days, 7 lags
        (15, ['A'], [1, 2, 3, 4, 5, 6, 7]),
    ]
)
def test_get_time_lags_dimensions(data_size, unique_ids, n_lags):
    # Create sample data
    data = {
        'ds': [f'2023-01-{i+1:02d}' for i in range(data_size)] * len(unique_ids),
        'unique_id': [id_ for id_ in unique_ids for _ in range(data_size)],
        'y': list(range(data_size * len(unique_ids)))  # Just sequential numbers
    }
    
    df = pl.DataFrame(data)
    
    # Execute function
    result_df = get_time_lags(df, lags=n_lags)
    result_with_nulls = get_time_lags(df, lags=n_lags, drop_nulls=False)
    
    # Test 1: Check number of rows after removing nulls
    expected_rows = data_size - max(n_lags)
    assert result_df.shape[0] == expected_rows * len(unique_ids)
    
    # Test 2: Check number of columns
    expected_cols = 3 + len(n_lags)  # Original columns (ds, unique_id, y) + lag columns
    assert len(result_df.columns) == expected_cols
    assert len(result_with_nulls.columns) == expected_cols
    
    # Additional verification that all lag columns are present
    lag_columns = [f'y__{lag}__lag' for lag in n_lags]
    assert all(col in result_df.columns for col in lag_columns)



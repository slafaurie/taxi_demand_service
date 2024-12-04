import pytest 
from datetime import date
from src.etl.helpers import generate_list_of_months

def test_calculate_months_diff():
    from_date = date(2023,9,5)
    to_date = date(2024,3,28)
    
    result = generate_list_of_months(from_date, to_date)
    
    expected = [
        date(2023,9,1),
        date(2023,10,1),
        date(2023,11,1),
        date(2023,12,1),
        date(2024,1,1),
        date(2024,2,1),
        date(2024,3,1),
    ]
    
    assert result == expected
    
    

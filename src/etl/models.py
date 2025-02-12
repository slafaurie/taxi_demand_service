import polars as pl 


class NYCPickupHourlySchema:
    """This class defines the types

    Returns:
        _type_: _description_
    """
    
    SCHEMA = [
        {"column": "key", "type": pl.String},
        {"column": "pickup_datetime_hour", "type": pl.Datetime},
        {"column": "num_pickup", "type": pl.Int32},
        {"column": "pickup_location_id", "type": pl.Int32}
    ]
    
    @classmethod
    def _get_columns(cls) -> list:
        return [x.get('column') for x in cls.SCHEMA]
    
    @classmethod
    def _get_type_mapping(cls) -> dict:
        return { x.get('column'):x.get('type') for x in cls.SCHEMA }  
    
    @classmethod
    def enforce_schema(cls, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df 
            .select(cls._get_columns())
            .cast(cls._get_type_mapping())
        )
        
import great_expectations as ge
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DataExpectations:
    """Great Expectations for data validation"""
    
    def __init__(self, suite_name: str = "real_estate_suite"):
        self.suite_name = suite_name
        self.suite = ExpectationSuite(expectation_suite_name=suite_name)
        self._create_expectations()
    
    def _create_expectations(self):
        """Create expectation suite"""
        
        # Column existence expectations
        required_columns = ['property_id', 'square_feet', 'bedrooms', 
                           'bathrooms', 'year_built', 'price']
        
        for col in required_columns:
            self.suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": col}
                )
            )
        
        # Numeric range expectations
        self.suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "square_feet",
                    "min_value": 100,
                    "max_value": 50000
                }
            )
        )
        
        self.suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "bedrooms",
                    "min_value": 0,
                    "max_value": 10
                }
            )
        )
        
        self.suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "bathrooms",
                    "min_value": 0,
                    "max_value": 10
                }
            )
        )
        
        self.suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "year_built",
                    "min_value": 1800,
                    "max_value": 2024
                }
            )
        )
        
        self.suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "price",
                    "min_value": 10000,
                    "max_value": 10000000
                }
            )
        )
        
        # Unique constraints
        self.suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "property_id"}
            )
        )
        
        # Not null expectations
        for col in required_columns:
            self.suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col}
                )
            )
        
        logger.info(f"Created expectation suite with {len(self.suite.expectations)} expectations")
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe against expectations"""
        
        ge_df = ge.from_pandas(df)
        results = ge_df.validate(expectation_suite=self.suite)
        
        success = results["success"]
        stats = results["statistics"]
        
        logger.info(f"Validation {'passed' if success else 'failed'}")
        logger.info(f"Success rate: {stats['success_percent']:.2f}%")
        
        if not success:
            failed_expectations = [
                r for r in results["results"] if not r["success"]
            ]
            for failure in failed_expectations[:5]:  # Log first 5 failures
                logger.warning(f"Failed: {failure['expectation_config']['expectation_type']}")
        
        return {
            "success": success,
            "success_percent": stats["success_percent"],
            "evaluated_expectations": stats["evaluated_expectations"],
            "failed_expectations": stats["failed_expectations"],
            "details": results["results"]
        }
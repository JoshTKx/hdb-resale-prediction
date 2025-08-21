# HDB Resale Price Dataset

## Dataset Overview

This folder contains the HDB (Housing Development Board) resale flat prices dataset for Singapore, covering transactions from 2017 to present.

## Data Source

- **Original Source**: Data.gov.sg - Singapore Government's Open Data Portal
- **Dataset**: Resale Flat Prices
- **URL**: https://data.gov.sg/dataset/resale-flat-prices
- **License**: Singapore Open Data License

## Dataset Description

### File Structure
- `sample_hdb_data.csv` - Sample of 1,000 transactions for testing/demonstration
- Full dataset (213,883 records) available from original source

### Data Schema

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `month` | String | Transaction year-month | "2017-01" |
| `town` | String | HDB town name | "ANG MO KIO" |
| `flat_type` | String | Type of flat | "3 ROOM", "4 ROOM", "EXECUTIVE" |
| `block` | String | Block number | "174", "15A" |
| `street_name` | String | Street name | "ANG MO KIO AVE 4" |
| `storey_range` | String | Floor level range | "04 TO 06", "10 TO 12" |
| `floor_area_sqm` | Float | Floor area in square meters | 73.0, 123.0 |
| `flat_model` | String | HDB flat model | "Improved", "Premium Maisonette" |
| `lease_commence_date` | Integer | Year lease started | 1986, 2010 |
| `remaining_lease` | String | Remaining lease duration | "61 years 04 months" |
| `resale_price` | Float | Transaction price in SGD | 350000.0, 680000.0 |

### Data Quality

- **Completeness**: No missing values across all columns
- **Consistency**: Standardized categorical values
- **Temporal Coverage**: January 2017 to present
- **Geographic Coverage**: All 26 HDB towns in Singapore
- **Transaction Volume**: 213,883 verified resale transactions

### Key Statistics

- **Price Range**: SGD 140,000 - SGD 1,660,000
- **Average Price**: SGD 519,000
- **Median Price**: SGD 488,000
- **Average Floor Area**: 97 sqm
- **Most Common Flat Type**: 4 ROOM (45% of transactions)
- **Towns Covered**: 26 (complete coverage)
- **Flat Types**: 7 categories
- **Time Period**: 8+ years of data

## Data Usage Notes

### For Model Training
- Use time-based splits (avoid random splits)
- Recommended split: 2017-2022 (train), 2023 (validation), 2024+ (test)
- Consider seasonal patterns when modeling

### Feature Engineering Considerations
- `remaining_lease` requires text parsing (years + months → total months)
- `storey_range` can be converted to numerical ranges
- `month` should be split into year and month components
- `lease_commence_date` can derive building age

### Categorical Variables
- **High Cardinality**: `block` (2,743 unique), `street_name` (1,400+ unique)
- **Suitable for Embeddings**: `town`, `flat_type`, `storey_range`, `flat_model`
- **Geographic Hierarchy**: Town → Street → Block

## Preprocessing Pipeline

The data undergoes the following transformations:

1. **Temporal Features**: Extract year and month from `month` field
2. **Text Parsing**: Convert `remaining_lease` to numerical months
3. **Age Calculation**: Derive building age from lease commence date
4. **Categorical Encoding**: Label encoding for embedding features
5. **Normalization**: StandardScaler for continuous variables
6. **Time-based Split**: Chronological train/validation/test split

## Data Ethics & Privacy

- **Public Dataset**: All data is publicly available government data
- **Anonymized**: No personal information included
- **Aggregated**: Transaction-level data without individual identifiers
- **Research Use**: Suitable for academic and research purposes

## Contact

For questions about data processing or feature engineering, refer to the preprocessing notebook or open an issue in the repository.
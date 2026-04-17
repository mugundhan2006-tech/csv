import pandas as pd


def load_sales_data(path: str) -> pd.DataFrame:
    """Load sales CSV data into a DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def summarize_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic sales summaries."""
    summary = {
        'rows': len(df),
        'total_revenue': df['Revenue'].sum() if 'Revenue' in df else None,
        'average_order_value': df['Revenue'].mean() if 'Revenue' in df else None,
        'unique_customers': df['CustomerID'].nunique() if 'CustomerID' in df else None,
        'unique_products': df['Product'].nunique() if 'Product' in df else None,
    }
    return pd.DataFrame([summary])


def product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and quantity by product."""
    if 'Product' not in df or 'Revenue' not in df:
        raise ValueError('DataFrame must contain Product and Revenue columns.')
    return (
        df.groupby('Product', as_index=False)
          .agg(total_revenue=('Revenue', 'sum'), total_quantity=('Quantity', 'sum'))
          .sort_values('total_revenue', ascending=False)
    )


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by region."""
    if 'Region' not in df or 'Revenue' not in df:
        raise ValueError('DataFrame must contain Region and Revenue columns.')
    return df.groupby('Region', as_index=False).agg(total_revenue=('Revenue', 'sum')).sort_values('total_revenue', ascending=False)


def monthly_sales(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """Aggregate revenue by month."""
    if date_col not in df:
        raise ValueError(f'Date column {date_col} not found in DataFrame.')
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().any():
        raise ValueError('Some dates could not be parsed. Check the date format.')
    df['Month'] = df[date_col].dt.to_period('M').astype(str)
    return df.groupby('Month', as_index=False).agg(total_revenue=('Revenue', 'sum')).sort_values('Month')


if __name__ == "__main__":
    # Load the sales data
    df = load_sales_data('sales_data.csv')
    
    # Print basic summary
    print("Sales Summary:")
    summary = summarize_sales(df)
    print(summary.to_string(index=False))
    print("\n")
    
    # Product performance
    print("Product Performance:")
    products = product_performance(df)
    print(products.to_string(index=False))
    print("\n")
    
    # Sales by region
    print("Sales by Region:")
    regions = sales_by_region(df)
    print(regions.to_string(index=False))
    print("\n")
    
    # Monthly sales
    print("Monthly Sales:")
    monthly = monthly_sales(df)
    print(monthly.to_string(index=False))


def main() -> None:
    csv_path = 'sales.csv'
    print(f'Loading sales data from {csv_path}')

    df = load_sales_data(csv_path)
    print('\n=== Sales Summary ===')
    print(summarize_sales(df).to_string(index=False))

    if {'Product', 'Revenue'}.issubset(df.columns):
        print('\n=== Product Performance ===')
        print(product_performance(df).head(10).to_string(index=False))

    if {'Region', 'Revenue'}.issubset(df.columns):
        print('\n=== Sales by Region ===')
        print(sales_by_region(df).to_string(index=False))

    if 'Date' in df.columns:
        print('\n=== Monthly Sales ===')
        print(monthly_sales(df).to_string(index=False))


if __name__ == '__main__':
    main()

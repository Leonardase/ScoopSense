import csv
from datetime import datetime
import pandas as pd

def load_data():
    """
    Creates the train and trainNeg CSVs using historical inventory and production levels.
    The original data is spare so we need to average things out to get data for every day.
    """
    # Load data
    inventory = pd.read_csv('data/temp/inventory.csv', parse_dates=['Date'], date_format='%m/%d/%y')
    inventory = inventory.dropna(subset=['Date'])
    production = pd.read_csv('data/temp/production.csv', parse_dates=['Date'], date_format='%m/%d/%y')

    # Melt to long format
    inv_long = inventory.melt(id_vars='Date', var_name='Flavor', value_name='Inventory')
    prod_long = production.melt(id_vars='Date', var_name='Flavor', value_name='Produced')

    # Create complete date range
    all_dates = pd.date_range(start=inventory['Date'].min(), end=inventory['Date'].max(), freq='D')
    all_flavors = inv_long['Flavor'].unique()

    # Create complete calendar
    calendar = pd.DataFrame(
        [(date, flavor) for date in all_dates for flavor in all_flavors],
        columns=['Date', 'Flavor']
    )

    # Merge inventory and production, allowing us to more easily see days without readings and calculate sales.
    df = calendar.merge(inv_long, on=['Date', 'Flavor'], how='left')
    df = df.merge(prod_long, on=['Date', 'Flavor'], how='left')

    # Mark actual inventory readings for finding the next valid date easier.
    df['Is_Actual_Reading'] = ~df['Inventory'].isna()

    # Forward-fill inventory for available stock calculation
    df['Filled_Inventory'] = df.groupby('Flavor')['Inventory'].ffill()
    df['Produced'] = df['Produced'].fillna(0)

    # Calculate available stock (inventory + production)
    df['Available'] = df['Filled_Inventory'] + df['Produced']

    df = df.groupby('Flavor', group_keys=False).apply(get_next_inventory_dates)

    # Merge with next inventory values
    df = df.merge(
        inv_long.rename(columns={'Date': 'Next_Inventory_Date', 'Inventory': 'Next_Inventory'}),
        on=['Next_Inventory_Date', 'Flavor'],
        how='left'
    )

    # Forward-fill Next_Inventory to handle cases without gaps
    df['Next_Inventory'] = df.groupby('Flavor')['Next_Inventory'].ffill()

    # Calculate cumulative production between inventory readings
    df['Cumulative_Produced'] = df.groupby(['Flavor', 'Next_Inventory_Date'])['Produced'].cumsum()

    # Calculate total production in each period
    df['Total_Produced_In_Period'] = df.groupby(['Flavor', 'Next_Inventory_Date'])['Produced'].transform('sum')

    # Now calculate sales: (Start Inventory + Total Produced) - End Inventory
    Start_Inventory = df.groupby(['Flavor', 'Next_Inventory_Date'])['Filled_Inventory'].transform('first')
    df['Total_Sales'] = (Start_Inventory + df['Total_Produced_In_Period']) - df['Next_Inventory']

    # Handle negative sales case
    df.loc[df['Total_Sales'] < 0, 'Total_Sales'] += df['Total_Produced_In_Period']

    # Calculate days in period
    df['Days_Between_Readings'] = df.groupby(['Flavor', 'Next_Inventory_Date'])['Date'].transform('count')

    # Calculate daily sales per day of period
    df['Sales'] = (df['Total_Sales'] / df['Days_Between_Readings']).clip(lower=0)

    # Define date ranges based on the most recent completed season
    train_start, train_end = get_dates_from_csv('data/temp/inventory.csv')

    # Apply adjustments to complete dataset
    dfNeg = adjust_sales_data(df)
    df = adjust_data(df)

    # Filter datasets
    train_df = df[df['Date'].between(train_start, train_end)]
    trainNeg = dfNeg[dfNeg['Date'].between(train_start, train_end)]

    # Pivot back to wide for viewing and modeling
    train_wide = train_df.pivot(index='Date', columns='Flavor', values='Adjusted_Sales')
    trainNeg_wide = trainNeg.pivot(index='Date', columns='Flavor', values='Adjusted_Sales')

    # Add metadata
    for df_wide in [train_wide, trainNeg_wide]:
        df_wide['Total_Sales'] = df_wide.where(df_wide >= 0, 0).sum(axis=1)
        df_wide['Day_of_Week'] = df_wide.index.day_name()
        df_wide['Day_of_Year'] = df_wide.index.dayofyear
        df_wide['Is_Weekend'] = df_wide.index.dayofweek >= 5
        df_wide['Num_Available_Flavors'] = df_wide.index.map(
            df.groupby('Date')['Num_Available_Flavors'].first()
        )

    # Save files
    format_csv(train_wide).to_csv(f'data/temp/train.csv')
    format_csv(trainNeg_wide).to_csv(f'data/temp/trainNeg.csv')

def count_available_flavors(df):
    """
    Count flavors with inventory > 0 or production > 0 on each date.
    """
    return (((df['Filled_Inventory'] > 0) | (df['Produced'] > 0))
            .groupby(df['Date'])
            .sum()
            .rename('Num_Available_Flavors'))

def get_next_inventory_dates(group):
    """
    Finds the next actual inventory reading date for each gap to help estimate sales.
    """
    reading_dates = group.loc[group['Is_Actual_Reading'], 'Date'].sort_values()

    group['Next_Inventory_Date'] = pd.NaT

    for reading_date in reading_dates:
        # For all dates before this reading_date, this is their next reading
        mask = (group['Date'] < reading_date) & (group['Next_Inventory_Date'].isna())
        group.loc[mask, 'Next_Inventory_Date'] = reading_date

    return group

def adjust_sales_data(df):
    """
    Changes value to -1 from zero if there were no sales and the flavor was potentially not available for purchase.
    Also adds a Num_Available_Flavor column to the dataframe.
    """
    # Count available flavors
    available_flavors = count_available_flavors(df)
    df = df.merge(available_flavors, left_on='Date', right_index=True, how='left')

    # Create adjusted sales column
    df['Adjusted_Sales'] = df['Sales']

    # Rule 1: When 29+ flavors available, set all zeros to -1
    high_availability_mask = (df['Num_Available_Flavors'] >= 29)
    df.loc[high_availability_mask & (df['Sales'] < 0.1), 'Adjusted_Sales'] = -1

    # Rule 2: When inventory is 0 and sales is 0, set to -1
    zero_inventory_mask = (df['Filled_Inventory'] == 0)
    df.loc[zero_inventory_mask & (df['Sales'] < 0.1), 'Adjusted_Sales'] = -1

    return df

def format_csv(df_wide):
    """
    Formats dataframe columns into CSV format.
    """
    df = df_wide.round(2)
    cols = (['Day_of_Week', 'Total_Sales', 'Day_of_Year', 'Is_Weekend', 'Num_Available_Flavors'] +
            [col for col in df.columns if col not in ['Day_of_Week', 'Total_Sales', 'Day_of_Year',
                                                      'Is_Weekend', 'Num_Available_Flavors']])
    return df[cols]

def adjust_data(df):
    """
    Changes Sales to Adjusted_Sales column to mimic the other dataset.
    """
    # Count available flavors
    available_flavors = count_available_flavors(df)
    df = df.merge(available_flavors, left_on='Date', right_index=True, how='left')

    # Create adjusted sales column
    df['Adjusted_Sales'] = df['Sales'].where(df['Sales'] >= 0.1, 0)

    return df

def get_dates_from_csv(filename):
    """
    Gets the date range for training our model from a CSV file.
    Goes from the oldest date to the last date of the most recent completed season.
    """
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Extract dates from first column (skip header if exists)
    dates = []
    for row in rows[1:]:  # skip header row
        if row:  # skip empty rows
            dates.append(row[0])

    if not dates:
        return None, None

    # Convert to datetime objects
    date_objects = []
    for date in dates:
        try:
            date_obj = datetime.strptime(date, '%m/%d/%y')
            date_objects.append(date_obj)
        except ValueError:
            continue

    if not date_objects:
        return None, None

    # Get first date
    first_date = min(date_objects)

    # Get current year
    current_year = datetime.now().year

    # Filter dates from previous year
    prev_year_dates = [d for d in date_objects if d.year == current_year - 1]

    # Get last date from previous year
    last_date_prev_year = max(prev_year_dates) if prev_year_dates else None

    return (first_date.strftime('%m/%d/%Y'),
            last_date_prev_year.strftime('%m/%d/%Y')) if last_date_prev_year else None

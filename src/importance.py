import random
import pandas as pd
from datetime import datetime, timedelta
from ast import literal_eval
from itertools import combinations
from init import load_data
from predict import predict


def calculate_prioritized_runout_dates(input_date_str):
    """
    Calculates runout dates with advanced prioritization based on these rules:
    - Importance 1 flavors must never run out
    - Maintain minimum 29 flavors in stock
    - Always keep at least 1 sorbet available
    - Prioritize connected flavors for production efficiency

    Allows us to map a data based production list that follows both inventory needs and rinse cycles.
    Also allows insight as to when to expect a flavor to run out.
    """
    print("[3/8] Formating data...")
    load_data()
    print("[4/8] Getting dates...")
    get_dates(input_date_str)

    predict()

    try:
        # Load and prepare all datasets
        sub = []
        need = []
        input_date = datetime.strptime(input_date_str, '%m-%d-%Y')

        # Load inventory and find most recent record
        inventory = pd.read_csv(f'data/temp/inventory.csv')
        inventory['Date'] = pd.to_datetime(inventory['Date'], format='%m/%d/%y')
        latest_inventory = inventory[inventory['Date'] <= input_date].sort_values('Date').iloc[-1]

        # Load predictions
        predictions = pd.read_csv(f'data/temp/predictions.csv')
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        future_predictions = predictions[predictions['Date'] >= input_date].sort_values('Date')

        # Load and parse flavor metadata
        metadata = pd.read_csv(f'data/temp/flavor_weights.csv')
        metadata = metadata.set_index(metadata.columns[0]).transpose()
        metadata['Order'] = metadata['Order'].apply(lambda x: literal_eval(x) if pd.notna(x) else [])

        # Get all flavor columns
        all_flavors = [col for col in latest_inventory.index if col in predictions.columns and col in metadata.index]

        # Calculate baseline runout dates
        runout_data = []
        for flavor in all_flavors:
            current_inv = latest_inventory[flavor]
            if current_inv <= 0:
                # Creates a data point for flavors that we are already out of stock for
                runout_data.append({
                    'Flavor': flavor,
                    'Runout_Date': 'Already Out',
                    'Days_Remaining': 0,
                    'Starting_Inventory': 0
                })
                continue

            sales = 0
            days_remaining = 0
            runout_date = None

            # Calculates the runout date based on our predictions
            for _, row in future_predictions.iterrows():
                sales += row[flavor]
                days_remaining += 1
                if sales >= current_inv:
                    runout_date = row['Date']
                    break

            # Creates a data point for flavors that we have stock for
            runout_data.append({
                'Flavor': flavor,
                'Runout_Date': runout_date.strftime('%Y-%m-%d') if runout_date else 'Beyond Prediction',
                'Days_Remaining': days_remaining if runout_date else len(future_predictions),
                'Starting_Inventory': round(current_inv, 2)
            })

        runout_df = pd.DataFrame(runout_data)

        # Merge with metadata
        runout_df = runout_df.merge(
            metadata[['Important', 'Sorbet', 'Nut', 'Order']],
            left_on='Flavor',
            right_index=True
        )

        def calculate_priority(row):
            """
            Helper function that gives priority scores to the need of a flavor.
            If a flavor is important it gets 14 points minus the days remaining.
            There is also a bonus given to flavors that have flavors that
            can be made without rinsing and have few days remaining.
            """
            priority = 0

            # Rule 1: Important flavors must never run out
            if int(row['Important']) == 1:
                priority += 14 - int(row['Days_Remaining'])

            # Rule 2: Sorbet priority (we must always have at least 1)
            if 'Sorbet' in row['Flavor']:
                if sum('Sorbet' in col for col in row.index) == 1:
                    priority += 13 - int(row['Days_Remaining'])
                    row['Important'] = 1

            # Rule 3: Give priority to flavors that can be made with other flavors too.
            connected_bonus = 0
            for connected_flavor in row['Order']:
                if connected_flavor in all_flavors:
                    connected_flavor_data = runout_df[runout_df['Flavor'] == connected_flavor].iloc[0]
                    # Add bonus if connected flavor is also running out
                    if connected_flavor_data['Days_Remaining'] < 5:
                        connected_bonus += 1
            priority += connected_bonus

            # Rule 4: Minimum flavor count protection (29 flavors)
            active_flavors = runout_df[(~runout_df['Runout_Date'].isin(['Already Out']))].shape[0]
            need.append(29 - active_flavors)

            # If we don't have enough flavors we recommend the non-important flavors too.
            if active_flavors < 30:
                if int(row['Important']) == 0:  # Non-important flavors get priority to maintain count
                    priority = 14 - int(row['Days_Remaining'])
                    if row['Flavor'] not in sub:
                        sub.append(row['Flavor'])

            return priority

        runout_df['Priority_Score'] = runout_df.apply(calculate_priority, axis=1)

        # Sort by priority
        prioritized_df = runout_df.sort_values(
            by=['Priority_Score', 'Days_Remaining'],
            ascending=[False, True]
        )

        # Format final output
        final_columns = [
            'Flavor', 'Important', 'Sorbet', 'Nut', 'Starting_Inventory',
            'Runout_Date', 'Days_Remaining', 'Priority_Score'
        ]

        # Ten most important flavors based on priority scores
        topTen = [[row.Flavor, row.Runout_Date, row.Days_Remaining] for row in prioritized_df.itertuples() if int(row.Important) == 1][:10]
        # Possible sorbet flavors that need to be made
        sorbets = [[row.Flavor, row.Runout_Date, row.Days_Remaining] for row in prioritized_df.itertuples() if 'Sorbet' in row.Flavor and row.Priority_Score > 9]
        # Non-important flavors that should be considered to be made
        others = [[row.Flavor, row.Runout_Date, row.Days_Remaining] for row in prioritized_df.itertuples() if row.Flavor in sub and row.Priority_Score > 9]
        # Non-important flavors that we don't have any stock for
        more = [[row.Flavor, row.Runout_Date, row.Days_Remaining, row.Order] for row in prioritized_df.itertuples() if row.Flavor in sub and row.Priority_Score == 14]
        # Flavor names for the topTen most important flavors and sorbets
        selected_flavors = set(row[0] for row in topTen + sorbets)

        # Flavors to create our ordering on
        high = [
            [row.Flavor, row.Runout_Date, row.Days_Remaining, row.Order]
            for row in prioritized_df.itertuples()
            if row.Flavor in selected_flavors and row.Priority_Score > 10
        ] + more

        # Print statements to display important production information of above predicated arrays
        print('Top Ten Most Important Flavors:')
        for x in range(len(topTen)):
            print(f'{x+1}.\t{topTen[x][0]}')
        print()

        if len(sorbets) > 0:
            print('More of this Sorbet or other:')
            for x in range(len(sorbets)):
                print(f'{x+1}.\t{sorbets[x][0]}')
            print()

        if len(others) > 0:
            print('Other Flavors to watch out for:')
            for x in range(len(others)):
                print(f'{x + 1}.\t{others[x][0]}')
            print()

        if need[0] > 0:
            print(f"We need {need[0]} flavors from these or others to get to 29 flavors")
            for row in more:
                print(f'\t{row[0]}')
            print()
        else:
            need[0] = 0

        print("Recommended Order to maintain 29 flavors and our important flavors:")
        flavs = [row[0] for row in more]
        # Finds the most efficient production list
        most = generate_efficient_orders(high)
        while len(flavs) < need[0]:
            flavs.append('Other')
        combos = list(combinations(flavs, need[0]))

        # Breaks ties for the non-important flavors
        scored = []
        for combo in combos:
            score, cleaned = score_combo(most, set(combo), flavs)
            scored.append((score, combo, cleaned))

        # Get the combos with the lowest RINSE count
        min_score = min(s[0] for s in scored)
        best_options = [s for s in scored if s[0] == min_score]

        # Choose one at random if tied
        chosen = random.choice(best_options)
        final_order = chosen[2]

        for flavor in final_order:
            print(f'\t{flavor}')
        return prioritized_df[final_columns]

    except Exception as e:
        print(f"Error in prioritized runout calculation: {e}")
        return pd.DataFrame()

def get_dates(input_date_str):
    """
    Creates a CSV with 14 days of data from the previous year, filling missing dates with averages. This is used to test
    our model.
    """
    try:
        # Parse input date
        input_date = datetime.strptime(input_date_str, '%m-%d-%Y')
        input_year = input_date.year

        # Calculate date range from previous year
        prev_year_start = input_date.replace(year=input_year - 1)
        date_range = [prev_year_start + timedelta(days=i) for i in range(14)]

        # Read and prepare training data
        df = pd.read_csv(f'data/temp/train.csv')
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate average values for all flavor columns
        flavor_columns = [col for col in df.columns if col not in ['Date', 'Day_of_Week', 'Total_Sales', 'Day_of_Year', 'Is_Weekend']]
        avg_values = df[flavor_columns].mean().to_dict()

        # Create template for missing dates
        day_of_week_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Process each date in range
        result_rows = []
        for target_date in date_range:
            # Try to find matching data
            match = df[df['Date'] == target_date]

            if not match.empty:
                # Use existing data
                row = match.iloc[0].copy()
                row['Date'] = row['Date'].replace(year=input_year)
            else:
                # Create average row for missing data as a backup
                row = {
                    'Date': target_date.replace(year=input_year),
                    'Day_of_Week': day_of_week_map[target_date.weekday()],
                    'Total_Sales': round(df['Total_Sales'].mean(), 2),
                    'Day_of_Year': target_date.timetuple().tm_yday,
                    'Is_Weekend': target_date.weekday() >= 5,
                }
                for flavor, avg_value in avg_values.items():
                    row[flavor] = round(avg_value, 2)

            result_rows.append(row)

        # Create final DataFrame
        result_df = pd.DataFrame(result_rows)

        # Format date consistently
        result_df['Date'] = result_df['Date'].dt.strftime('%Y-%m-%d')

        # Save results
        result_df.to_csv(f'data/temp/output.csv', index=False)

        return 'output.csv'

    except Exception as e:
        print(f"Error: {e}")
        return False

def _classify_flavors(high, meta):
    """
    Classifies flavors into tiers based on metadata.
    We want to make the nuts flavors last for allergy reasons, and chocolate second to last for efficiency.
    """
    tier1, tier2, tier3 = [], [], []

    for row in high:
        flavor = row[0]
        nut = int(meta.at[flavor, 'Nut']) if flavor in meta.index else 0
        choc = int(meta.at[flavor, 'Chocolate']) if flavor in meta.index else 0

        if nut == 1:
            tier3.append(row)
        elif choc == 1:
            tier2.append(row)
        else:
            tier1.append(row)

    return tier1, tier2, tier3

def _build_flavor_graph(tier_rows):
    """
    Builds a digraph graph for flavors in a tier.
    Allows us to map the flavors we can make without a rinse of the machine.
    """
    high_flavors = {row[0] for row in tier_rows}
    flavor_graph = {}

    for row in tier_rows:
        current_flavor = row[0]
        # Get compatible flavors that are also in high_flavors
        compatible = [f for f in row[3] if f in high_flavors and f != current_flavor]
        flavor_graph[current_flavor] = compatible

    return flavor_graph, high_flavors

def _find_longest_chain(start_flavor, flavor_graph):
    """
    Find the longest chain starting from a given flavor that avoids rinsing the machine.
    """
    def dfs(path, used_local):
        current = path[-1]
        extended = False

        for neighbor in flavor_graph.get(current, []):
            if neighbor not in used_local:
                dfs(path + [neighbor], used_local | {neighbor})
                extended = True

        if not extended:
            # Store the completed chain
            all_chains.append(path)

    all_chains = []
    dfs([start_flavor], {start_flavor})

    # Return the longest chain found
    return max(all_chains, key=len) if all_chains else [start_flavor]


def _build_chains_for_tier(tier_rows):
    """
    Build chains of flavors in their respective tier that can be made without rinsing the machine.
    """
    flavor_graph, high_flavors = _build_flavor_graph(tier_rows)
    used = set()
    full_order = []

    while len(used) < len(high_flavors):
        best_chains = []

        # Try starting from each unused flavor
        for flavor in high_flavors - used:
            chain = _find_longest_chain(flavor, flavor_graph)
            best_chains.append(chain)

        # Pick the longest chain
        longest = max(best_chains, key=len)
        full_order.extend(longest)
        used.update(longest)

        # Add RINSE if there are more flavors to process
        if len(used) < len(high_flavors):
            full_order.append("RINSE")

    return full_order


def generate_efficient_orders(high):
    """
    Generates the efficient order sequences for priority flavors
    that minimizes th number of machin rinses and maximizes flavor need.
    """
    # Load metadata to get Nut and Choc info
    meta = pd.read_csv('data/temp/flavor_weights.csv')
    meta = meta.set_index(meta.columns[0]).transpose()

    # Classify flavors into tiers
    tier1, tier2, tier3 = _classify_flavors(high, meta)

    # Combine all chains tier by tier
    final_order = []

    for tier in [tier1, tier2, tier3]:
        if tier:
            chain = _build_chains_for_tier(tier)
            final_order.extend(chain)
            final_order.append("RINSE")  # Add RINSE between tiers

    # Remove trailing RINSE if present
    if final_order and final_order[-1] == "RINSE":
        final_order.pop()

    return final_order


def score_combo(order, keep_set, maybe_flavors):
    """
    Finds the most efficient combination of flavor need by scoring a potential order,
    helping to minimize number of rinses
    """
    temp_order = [f for f in order if f not in maybe_flavors or f in keep_set]

    # Collapse consecutive Rinses and trim start/end if needed
    cleaned = []
    for item in temp_order:
        if item == "RINSE" and (not cleaned or cleaned[-1] == "RINSE"):
            continue
        cleaned.append(item)

    # Remove trailing RINSE
    while cleaned and cleaned[-1] == "RINSE":
        cleaned.pop()

    return cleaned.count("RINSE"), cleaned
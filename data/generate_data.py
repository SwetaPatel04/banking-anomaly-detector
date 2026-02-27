# Import numpy for numerical operations like generating realistic transaction amounts
import numpy as np
# Import pandas to organize data into a table (DataFrame) and save as CSV
import pandas as pd
# Import random for generating random choices like merchants and user behaviour
import random
# Import datetime tools to generate timestamps spread across multiple days
from datetime import datetime, timedelta

# List of normal, real-world merchants a typical Canadian bank customer would use
MERCHANTS = ["Tim Hortons", "Shoppers Drug Mart", "No Frills",
             "Netflix", "Amazon", "Uber", "Bell Canada", "Walmart"]

# List of suspicious merchants that would appear in fraudulent transactions
SHADY_MERCHANTS = ["Unknown Vendor #381", "INTL_TRANSFER_XYZ", "CryptoExchange99"]

# Main function to generate the full dataset
# n_users = how many fake bank customers to simulate
# n_days = how many days of transaction history to generate
# anomaly_rate = what fraction of transactions should be fraudulent (5% default)
def generate_dataset(n_users=50, n_days=60, anomaly_rate=0.05):

    # Empty list to collect all transaction records before turning into a DataFrame
    records = []

    # Set the start date as 60 days ago from today
    start = datetime.now() - timedelta(days=n_days)

    # Loop through each user (USER-0001, USER-0002, etc.)
    for i in range(1, n_users + 1):

        # Create a zero-padded user ID string like USER-0001
        user_id = f"USER-{str(i).zfill(4)}"

        # Give each user a random number of transactions between 20 and 80
        for _ in range(random.randint(20, 80)):

            # Pick a random day within our date range for this transaction
            date = start + timedelta(days=random.randint(0, n_days))

            # Randomly decide if this transaction is fraudulent based on anomaly_rate
            is_anomaly = random.random() < anomaly_rate

            # Build the transaction record as a dictionary
            records.append({
                # User who made the transaction
                "user_id": user_id,

                # Anomalous transactions have suspiciously high amounts (2000-9999)
                # Normal transactions follow a log-normal distribution (realistic everyday spending)
                "amount": round(random.uniform(2000, 9999), 2) if is_anomaly
                          else round(np.random.lognormal(3.5, 1.0), 2),

                # Fraudulent transactions happen at odd hours (2am-4am)
                # Normal transactions are weighted toward business hours (8am-6pm)
                "hour": random.randint(2, 4) if is_anomaly
                        else random.choices(range(24), weights=[1,1,1,1,1,2,4,8,10,10,10,10,10,10,10,10,10,10,8,6,5,4,3,2])[0],

                # 0 = unknown/shady merchant, 1 = known legitimate merchant
                "merchant_known": 0 if is_anomaly else 1,

                # 1 if the transaction happened on Saturday or Sunday, 0 otherwise
                "is_weekend": 1 if date.weekday() >= 5 else 0,

                # Label: 1 = fraud/anomaly, 0 = normal transaction
                "is_anomaly": int(is_anomaly)
            })

    # Convert the list of dictionaries into a pandas DataFrame (like a spreadsheet)
    df = pd.DataFrame(records)

    # Save the DataFrame to a CSV file in the root project folder
    df.to_csv("transactions.csv", index=False)

    # Print a summary so we can confirm the data looks right
    print(f"Generated {len(df)} transactions ({df['is_anomaly'].sum()} anomalies)")

    # Return the DataFrame in case we want to use it directly in another script
    return df

# This block runs only when you execute this file directly (not when imported)
if __name__ == "__main__":
    generate_dataset()
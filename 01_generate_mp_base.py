"""
Phase 1: Generate the MP base table (543 seats, real state-wise Lok Sabha
seat distribution). Mirrors the schema you already pulled from the real
MPLADS 'Allocated Limit for Hon'ble MPs' export: STATE | MP_NAME |
CONSTITUENCY | ALLOCATED_AMOUNT | TENURE_START | TENURE_END.

If you have the real CSV from mplads.mospi.gov.in, drop it in as
'real_allocated_limit.csv' and this script will use it instead of
generating names -- everything downstream (features/model) is schema-
compatible either way.
"""
import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

REAL_FILE = "real_allocated_limit.csv"

# Real Lok Sabha seat distribution by state/UT (543 total)
STATE_SEATS = {
    "Uttar Pradesh": 80, "Maharashtra": 48, "West Bengal": 42, "Bihar": 40,
    "Tamil Nadu": 39, "Madhya Pradesh": 29, "Karnataka": 28, "Gujarat": 26,
    "Rajasthan": 25, "Andhra Pradesh": 25, "Odisha": 21, "Kerala": 20,
    "Telangana": 17, "Jharkhand": 14, "Assam": 14, "Punjab": 13,
    "Chhattisgarh": 11, "Haryana": 10, "Delhi": 7, "Jammu and Kashmir": 5,
    "Uttarakhand": 5, "Himachal Pradesh": 4, "Goa": 2, "Tripura": 2,
    "Meghalaya": 2, "Manipur": 2, "Arunachal Pradesh": 2, "Nagaland": 1,
    "Mizoram": 1, "Sikkim": 1, "Chandigarh": 1, "Puducherry": 1,
    "Andaman and Nicobar Islands": 1, "Dadra and Nagar Haveli and Daman and Diu": 2,
    "Lakshadweep": 1, "Ladakh": 1,
}
assert sum(STATE_SEATS.values()) == 543

FIRST_NAMES = ["Raj", "Amit", "Suresh", "Anil", "Vikram", "Ravi", "Sanjay",
    "Deepak", "Manoj", "Ashok", "Sunita", "Priya", "Kavita", "Meena",
    "Geeta", "Rekha", "Anita", "Poonam", "Rajesh", "Mahesh", "Naresh",
    "Dinesh", "Prakash", "Ramesh", "Vinod", "Ajay", "Sanjeev", "Rakesh",
    "Yogesh", "Satish", "Usha", "Kiran", "Lata", "Shobha", "Neelam",
    "Arun", "Ashish", "Vijay", "Kishore", "Harish"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Reddy", "Singh", "Kumar", "Yadav",
    "Gupta", "Nair", "Rao", "Chauhan", "Mehta", "Joshi", "Choudhary",
    "Das", "Roy", "Iyer", "Pillai", "Naidu", "Thakur", "Pandey",
    "Mishra", "Bhatt", "Rathore", "Gaikwad", "Deshmukh", "Kulkarni",
    "Bhattacharya", "Chatterjee", "Banerjee", "Mondal", "Barman"]

def synth_name(used):
    while True:
        n = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if n not in used:
            used.add(n)
            return n

def build_synthetic():
    rows, used_names = [], set()
    mp_id = 1
    for state, seats in STATE_SEATS.items():
        for i in range(seats):
            constituency = f"{state} PC-{i+1}"
            allocated = 500000000  # Rs 5 Cr/annum, current MPLADS norm
            rows.append({
                "MP_ID": f"MP{mp_id:04d}",
                "STATE": state,
                "MP_NAME": synth_name(used_names),
                "CONSTITUENCY": constituency,
                "ALLOCATED_AMOUNT": allocated,
                "TENURE_START_DATE": "2024-06-01",
                "TENURE_END_DATE": "2029-05-31",
            })
            mp_id += 1
    return pd.DataFrame(rows)

if os.path.exists(REAL_FILE):
    print(f"Found {REAL_FILE} -- using real MoSPI export as the MP base.")
    df = pd.read_csv(REAL_FILE)
else:
    print("No real export found -- generating a realistic synthetic MP base "
          "(real state/seat distribution, placeholder names).")
    df = build_synthetic()

df.to_csv("mp_base.csv", index=False)
print(f"Wrote mp_base.csv -- {len(df)} MPs across {df['STATE'].nunique()} states/UTs")
print(df.head())

"""Sample application with intentional issues for the code review agent to find."""

import os
import pickle
import sqlite3
import subprocess


DB_PASSWORD = "super_secret_password_123"


def getUserData(user_id: str):
    """Fetch user from database."""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = '{user_id}'")
    result = cursor.fetchone()
    conn.close()
    return result


def process_upload(filename: str, data: bytes):
    obj = pickle.loads(data)
    path = os.path.join("/uploads", filename)
    with open(path, "wb") as f:
        f.write(obj)
    return path


def run_report(report_name: str):
    os.system(f"python reports/{report_name}.py")


def calculate_discount(price, qty, customer_type, region, day_of_week, coupon_code):
    """Calculate final price with discount."""
    if customer_type == "premium":
        if region == "US":
            if day_of_week in ("Saturday", "Sunday"):
                if coupon_code:
                    if coupon_code.startswith("SUPER"):
                        discount = 0.4
                    elif coupon_code.startswith("HALF"):
                        discount = 0.5
                    else:
                        discount = 0.25
                else:
                    discount = 0.2
            else:
                if coupon_code:
                    discount = 0.15
                else:
                    discount = 0.1
        elif region == "EU":
            if day_of_week in ("Saturday", "Sunday"):
                discount = 0.18
            else:
                discount = 0.12
        else:
            discount = 0.08
    elif customer_type == "regular":
        if coupon_code:
            discount = 0.1
        else:
            discount = 0.05
    else:
        discount = 0

    total = price * qty * (1 - discount)
    if total > 1000:
        total = total * 0.98
    if total > 5000:
        total = total * 0.97
    return total


class userData:
    """Holds user info."""

    def __init__(self, n, e, a):
        self.n = n
        self.e = e
        self.a = a

    def getFullName(self):
        return self.n

    def IsActive(self):
        return self.a

# src/resume/__init__.py
import sys
import pysqlite3

# Monkey-patch sqlite3 before ANYTHING imports it
sys.modules["sqlite3"] = pysqlite3
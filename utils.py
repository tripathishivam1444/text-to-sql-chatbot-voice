import pandas as pd
import os
from sqlalchemy import create_engine, inspect, text
from langchain.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()

DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "123456789")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "travel_db")
TABLE_NAME = "indian_travel_table"

# OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def get_db_engine():
    """Create and return a database engine"""
    try:
        connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

engine = get_db_engine()

def table_exists():
    """Check if the table already exists in the database"""
    try:
        inspector = inspect(engine)
        return TABLE_NAME in inspector.get_table_names()
    except Exception as e:
        st.error(f"Error checking table existence: {str(e)}")
        return False

def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    try:
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}")
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        return True
    except Exception as e:
        st.error(f"Failed to create database: {str(e)}")
        return False

def upload_to_database(df):
    """Upload dataframe to the database if it doesn't exist"""
    try:
        if table_exists():
            return True, "Data already exists in the database."

        if not create_database_if_not_exists():
            return False, "Failed to create database."

        df.columns = [col.strip() for col in df.columns]
        df.to_sql(TABLE_NAME, con=engine, if_exists='replace', index=False)

        if table_exists():
            row_count = engine.connect().execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).fetchone()[0]
            return True, f"Data successfully uploaded to {DB_NAME}.{TABLE_NAME} ({row_count} records)."
        else:
            return False, "Failed to create table."

    except Exception as e:
        return False, f"Error uploading to database: {str(e)}"

def clear_database():
    """Clear data from the database"""
    try:
        if not table_exists():
            return False, "Table does not exist."

        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
        return True, "Database data cleared successfully."
    except Exception as e:
        return False, f"Error clearing database: {str(e)}"

def create_Xsql_agent():
    """Create and return a SQL agent for querying the database"""
    try:
        if not table_exists():
            return None

        db = SQLDatabase.from_uri(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
        
        llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            max_tokens=8000,
            api_key=OPENAI_API_KEY
        )
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,

        )
        
        return agent_executor
    except Exception as e:
        st.error(f"Error creating SQL agent: {str(e)}")
        return None

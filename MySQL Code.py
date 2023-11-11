import pymysql
import pandas as pd
import mysql.connector
import os
import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, Float, update, text

# Passwords are wrong on purpose
# Database connection parameters
host_name = 'host_name'
db_name = 'database_name'
db_user = 'user'
db_password = 'password'

# Establish the database connection
connection = pymysql.connect(host=host_name,
                             user=db_user,
                             password=db_password,
                             database=db_name)

# SQL query to select the data
query = "SELECT * FROM your_table_name;"

# Execute the query and store the result in a dataframe
df = pd.read_sql(query, connection)

# Close the connection
connection.close()





# Replace these with your database credentials
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "world"
}

connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

table_name = "world_data"
date_column = "quote_datetime"

count_query = f"SELECT YEAR({date_column}), MONTH({date_column}), DAY({date_column}), COUNT(*) FROM {table_name} GROUP BY YEAR({date_column}), MONTH({date_column}), DAY({date_column})"

cursor.execute(count_query)
date_counts = cursor.fetchall()

for date, count in date_counts:
    print(f"Date: {date}, Count: {count}")

cursor.close()
connection.close()


# Uploading Excel files into MySQL with multiple cores
import os
import datetime
import pandas as pd
from sqlalchemy import create_engine
from multiprocessing import Pool, cpu_count


def import_excel_to_mysql(file_path):
    engine = create_engine('mysql+pymysql://root:1234@localhost:3306/world')
    data = pd.read_excel(file_path, header=0, engine='openpyxl')
    data.to_sql(name='world_data', con=engine, index=False, if_exists='append')
    return file_path


def main():
    start_time = datetime.datetime.now()
    print('Begin:', start_time)

    path = r'E:\Xlsx CBOE'
    files = os.listdir(path)

    # Leave one core free
    num_processes = cpu_count() - 3

    with Pool(num_processes) as pool:
        results = pool.map(import_excel_to_mysql, [os.path.join(path, i) for i in files])

    end_time = datetime.datetime.now()
    print('End', end_time)
    total_time = end_time - start_time
    print(total_time)
    print('Total number of imported files', len(results))


if __name__ == "__main__":
    main()



# Uploading Excel files into MySQL


start_time = datetime.datetime.now()
print('Begin:', start_time)
num = 0

engine = create_engine('mysql+pymysql://root:1234!@localhost:3306/world')
# charset=utf8mb4'
# connection = engine.connect()
path = r"C:"
files = os.listdir(path)

for i in files:
    data = pd.read_excel(os.path.join(path, i), header=0, engine='openpyxl')
    data.to_sql(name='test_data', con=engine, index=False, if_exists='append')
    num += 1
    print('imported:', i)

end_time = datetime.datetime.now()
print('End', end_time)
total_time = end_time - start_time
print(total_time)
print('Total number of imported files', num)


# Basic Code for MySQL

# USE world;                                                # MYSQL Code, Column is shown after doing running it twice
# SELECT * FROM world_trade;
# ALTER TABLE world_trade ADD Speed BIGINT;

# SELECT ask_size,                                           # MYSQL Code, for multiplying two columns and saving it in a new column
#        bid_size,
#        ask_size*bid_size AS Multipliziert
# FROM world_trade
# SELECT DISTINCT YEAR(quote_datetime), MONTH(quote_datetime), DAY(quote_datetime) FROM world.world_data
# SHOW BINARY LOGS;                       # Have to be deleted from time to time due to size
#
# PURGE BINARY LOGS TO 'X';
#
# SELECT DISTINCT YEAR(example), MONTH(example), DAY(example), FROM world.world_data
#
# SELECT DATE_FORMAT(order_date, '%y-%m-%d') AS formatted_date, COUNT(*) AS row_count
# FROM orders
# GROUP BY formatted_date
# ORDER BY formatted_date;
# connection.execute(text("ALTER TABLE world_trade ADD COLUMN Test_Testing varchar(255);"))
# def create_column(name):
#    connection.execute("ALTER TABLE world.world_trade ADD COLUMN %s varchar(100);" % (name))

# create_column('test')
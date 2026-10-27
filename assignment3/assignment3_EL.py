#Read the employee data from the csv file 
#PySpark RDD analysis of employee records

import csv
import os
import sys

from pyspark.sql import SparkSession

# Force Spark's workers to use the same Python that runs this script (conda's) - My Mac has Python 3.9 for OS and Python 3.11 and up for conda, and Spark was using 3.9, which caused errors when it tried to import spark.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
print("Spark will use Python at:", sys.executable)

#set up my data
from pathlib import Path
input_file = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).with_name("employee_records.csv")
)

if not input_file.is_file():
    raise FileNotFoundError(f"Input file not found: {input_file}")


spark = SparkSession.builder.appName("EmployeeDataRDD").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # hide Spark's INFO/WARN noise so the output is readable

 
def split_csv_line(line):
    """Split one CSV line into fields (handles quoted commas)."""
    return next(csv.reader([line]))
 
 
def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
 
 
def show(rdd_result):
    """Print a collected list of (key, value) pairs, one per line, sorted by key."""
    for key, value in sorted(rdd_result):
        print(f"{key}: {value}")

##1. Read the employee data from the csv file and create an RDD[String]
banner("1.Read the employee data from the csv file into RDD[String]")
lines = sc.textFile(str(input_file))
print("Type:", type(lines))
print("Total lines (incl. header):", lines.count())
print("First 3 lines:")
for line in lines.take(3):
    print("  ", line)

##2. Remove header with filter
banner("2.Remove the header line from the RDD[String] using RDD filters")
header = lines.first()
data = lines.filter(lambda line: line != header)
print("Total lines (excl. header):", data.count())
print("First 3 lines (excl. header):")
for line in data.take(3):
    print("  ", line)

# Locate the needed columns by name, so the code works whatever the column order is
columns = [c.strip().lower() for c in split_csv_line(header)]
DEPT_IDX = columns.index("department")
COUNTRY_IDX = columns.index("country")
SALARY_IDX = columns.index("salary")
 
 
def parse(line):
    fields = split_csv_line(line)
    return (
        fields[DEPT_IDX].strip(),
        fields[COUNTRY_IDX].strip(),
        float(fields[SALARY_IDX]),
    )
# (department, country, salary) tuples, cached because every section reuses them
records = data.map(parse).cache()

##3. Find the number of employees in each department using GroupByKey and ReduceByKey
banner("3.Find the number of employees in each department using GroupByKey and ReduceByKey")
dept_ones = records.map(lambda r: (r[0], 1))  # (department, 1) pairs

 
print("\n3.1 Using groupByKey():")
count_group = dept_ones.groupByKey().mapValues(lambda ones: len(list(ones)))
show(count_group.collect())
 
print("\n3.2 Using reduceByKey():")
count_reduce = dept_ones.reduceByKey(lambda a, b: a + b)
show(count_reduce.collect())

##4. Find the average salary per department using GroupByKey and ReduceByKey
banner("4.Find the average salary per department using GroupByKey and ReduceByKey")
### Create (department, salary) pairs
print("\n4.1 Using groupByKey():")
avg_dept_group = (
    records.map(lambda r: (r[0], r[2]))
    .groupByKey()
    .mapValues(lambda salaries: sum(salaries) / len(list(salaries)))
)

show([(k, round(v, 2)) for k, v in avg_dept_group.collect()])

print("\n4.2 Using reduceByKey():")
avg_dept_reduce = (
    records.map(lambda r: (r[0], (r[2], 1)))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .mapValues(lambda t: t[0] / t[1])
)
show([(k, round(v, 2)) for k, v in avg_dept_reduce.collect()])

##5. Find the average salary per country and department using GroupByKey and ReduceByKey
banner("5.Find the average salary per country and department using GroupByKey and ReduceByKey")
### Create ((country, department), salary) pairs
print("\n5.1 Using groupByKey():")
avg_country_dept_group = (
    records.map(lambda r: ((r[1], r[0]), r[2]))
    .groupByKey()
    .mapValues(lambda salaries: sum(salaries) / len(list(salaries)))
)
show([(k, round(v, 2)) for k, v in avg_country_dept_group.collect()])   

print("\n5.2 Using reduceByKey():")
avg_country_dept_reduce = (
    records.map(lambda r: ((r[1], r[0]), (r[2], 1)))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .mapValues(lambda t: t[0] / t[1])
)
show([(k, round(v, 2)) for k, v in avg_country_dept_reduce.collect()])

##6. Find the minimum and maximum salary per country by using GroupByKey and ReduceByKey
banner("6.Find the minimum and maximum salary per country by using GroupByKey and ReduceByKey")
### Create ((country), minimum salary, maximum salary) pairs
print("\n6.1 Using groupByKey(): country: (min_salary, max_salary)")
minmax_country_group = (
    records.map(lambda r: (r[1], r[2]))
    .groupByKey()
    .mapValues(lambda salaries: (min(salaries), max(salaries)))
)
show([(k, (round(v[0], 2), round(v[1], 2))) for k, v in minmax_country_group.collect()])  


print("\n6.2 Using reduceByKey(): country: (min_salary, max_salary)")
minmax_country_reduce = (
    records.map(lambda r: (r[1], (r[2], r[2])))
    .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1])))
)
show([(k, (round(v[0], 2), round(v[1], 2))) for k, v in minmax_country_reduce.collect()])

##7. Find the list of unique countries in the dataset and sort them in alphabetical order
banner("7.Find the list of unique countries in the dataset and sort them in alphabetical order")
countries = records.map(lambda r: r[1]).distinct().sortBy(lambda c: c)
for c in countries.collect():
    print(c)

spark.stop()
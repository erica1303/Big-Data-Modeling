
# Assignment 2 - Erica Luong: PySpark Warmup Using RDDs
# spark-submit --master 'local[*]' assignment_2_Erica_Luong.py temperatures.txt

import sys
import re
from pyspark.sql import SparkSession
 
int_pattern = re.compile(r"^[+-]?[0-9]+$")

def parse_record(line):
    # Split the line into pieces wherever there's a comma
    fields = line.split(",")

    # Reject anything without exactly 3 fields (this also catches blank lines)
    if len(fields) != 3:
        return []

    # Remove extra spaces around each field
    country = fields[0].strip()
    city = fields[1].strip()
    temp_text = fields[2].strip()

    # Reject the record if the temperature isn't a valid integer
    if not int_pattern.match(temp_text):
        return []

    # Valid record: return it as (country, city, integer temperature)
    return [(country, city, int(temp_text))]

def main():
    #require the input file as a command line argument
    if len(sys.argv) != 2:
        print("Usage: assignment_2_Erica_Luong.py <input_file>", file=sys.stderr)
        sys.exit(-1)

    input_file = sys.argv[1]

    spark = SparkSession.builder.appName("Assignment2").getOrCreate()
    sc = spark.sparkContext

    try:
        lines = sc.textFile(input_file)

        # Parse every line; invalid lines return [] and disappear
        records = lines.flatMap(parse_record)
        records.cache()

   # City averages: carry (sum, count) per city, combine, then divide
        city_pairs = records.map(lambda r: (r[1], (r[2], 1)))
        city_totals = city_pairs.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        city_averages = city_totals.mapValues(lambda t: t[0] / t[1])
        # Unique countries: take the country from each valid record, then remove duplicates
        countries = records.map(lambda r: r[0]).distinct()

         # Bring only the final results to the driver for display
        city_results = sorted(city_averages.collect())
        country_results = sorted(countries.collect())

        for city, avg in city_results:
            print((city, avg))
        print(("LIST_of_COUNTRIES", country_results))
    finally:
        # Always shut Spark down, even if something goes wrong
        spark.stop()

if __name__ == "__main__":
    main()






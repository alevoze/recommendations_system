import psycopg2
conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=alevoze")
cur = conn.cursor()
cur.execute(
"""
    CREATE TABLE recommendations(
    id integer PRIMARY KEY,
    result_rec text
)
"""
)

# insert_query = "INSERT INTO recommendations VALUES {}".format()
# cur.execute(insert_query)
conn.commit()
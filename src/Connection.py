import psycopg2 as p

def get_connection():
    # make connection
    try:
        connect = p.connect("dbname='dbname' user='classificator' host=xx.xx.xx.xx password='dd2f4fg2rtr57y4r2t'")
    except:
        print("Can't connect to the database")
    # cursor
    cur = connect.cursor()
    return cur, connect

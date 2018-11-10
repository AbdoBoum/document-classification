import psycopg2 as p


def get_connection():
    # make connection
    try:
        connect = p.connect("dbname='imperiumdb' user='classificator' host=192.168.3.23 password='dd2f4fg2rtr57y4r2t'")
    except:
        print("Can't connect to the database")
    # cursor
    cur = connect.cursor()
    return cur, connect

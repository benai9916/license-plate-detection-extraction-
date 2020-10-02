# Database
import sqlite3
conn = sqlite3.connect('licenseDetail.db',  check_same_thread=False)
c = conn.cursor()

# Functions
def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS licenseNoTable(licenseNo TEXT, complain TEXT, dates TEXT)')


def add_data(licenseNo,complain, dates):
	c.execute('INSERT INTO licenseNoTable(licenseNo,complain, dates) VALUES (?,?,?)',(licenseNo,complain, dates))
	conn.commit()
	return True

def fetch_data(licenseNo):
	c.execute('SELECT * FROM licenseNoTable WHERE licenseNo =?',(licenseNo,))
	data = c.fetchall()
	return data


def view_all_data():
	c.execute('SELECT * FROM licenseNoTable')
	data = c.fetchall()
	return data
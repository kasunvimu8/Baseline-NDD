
import sqlite3



###########################################################################
## string constants declaration ############
###########################################################################

conn = 'noconnyet'
cursor = 'nocursoryet'


###########################################################################
## SQLITE OPERATIONS ############
#####_######################################################################

def connectToDB(db):
	global conn
	global cursor

	if(conn=='noconnyet' or cursor =='nocursoryet'):
		print("CONNECTING TO DB ")
		try:
			conn= sqlite3.connect(db)
			cursor = conn.cursor()
			print('SUCCESSFULLY CONNECTED TO : ' + db)
		except Exception as e:
			print("ERROR CONNECTING TO DB !!!")
			print(e)
	return conn, cursor



def closeDBConnection():
	global conn
	if(conn == 'noconnyet'):
		print("NO DB CONNECTION ACTIVE")
	else:
		try:
			conn.commit()
			conn.close()
			print('SUCCESFULLY CLOSED DB CONNECTION')
			conn="noconnyet"
			cursor="nocursoryet"
		except:
			print("ERROR !! COULD NOT CLOSE CONNECTION")


def fetchAllNearDuplicates(condition=""):
	global cursor
	fetchNDStatement = ('''SELECT * FROM nearduplicates {0}''').format(condition)
	#print(checkCrawlEntryStatement)
	try: 
		cursor.execute(fetchNDStatement)
		allNDEntries = cursor.fetchall()
		return allNDEntries
	except Exception as e:
		print("ERROR ALL NEARDUPLICATES : " + condition)	
		print(e)	
		return None	

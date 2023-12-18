import pyodbc
from datetime import datetime
from threading import Thread

# conn = pyodbc.connect('Driver={};'
#                       'Server=localhost;'
#                       'Database=humanResource;'
#                       'Trusted_connection=yes;')

# cursor = conn.cursor()

class SqlQueries:
    def __init__(self, userId):
        self.conn = self.connection()
        self.cursor = self.conn.cursor()
        self.User = userId
        
        Thread(target=self.CheckAttandance, daemon=True ,args=(self.User)).start()

    def connection(self):
        return pyodbc.connect('Driver={SQL Server};'
                              'Server=localhost;'
                              'Database=humanResource;'
                              'UID=sa;'
                              'PWD=1234512345;'
                              'Trusted_connection=yes;')
    
        # return pyodbc.connect(Driver="{SQL Server}",
        #                 Server=localhost,
        #                 Database=humanResource,
        #                 UID=sa,
        #                 PWD=1234512345,
        #                 Trusted_connection=yes)


    def insertQuery(self, userId):
        # self.cursor.execute("Insert into 'table' (column1, column2) values (1 , 2) " (value1, value2))
        # self.cursor.execute("insert into Attendance(personalCode, [Date], EntranceTime) values (?, ?, ?) ", userId, 'cast(GETDATE() as date)', 'cast(getdate() as time)')
        # Row = (userId, 'cast(GETDATE() as date)', 'cast(getdate() as time)')
        self.cursor.execute("insert into Attendance(personalCode, [Date], EntranceTime) values (?, cast(GETDATE() as date), cast(getdate() as time) ) ", int(userId) ).commit()
        print('I do not result , just print userId', userId )
        # self.conn.commit()

    
    def ReadQuery(self):
        self.cursor.execute("select top 1 * from Attendance ")
        row = self.cursor.fetchall()
        print(row)
        return row
        
    def existQuery(self, userId):
        self.cursor.execute(
            '''
                declare @Result  int = 0
                if exists (select (personalCode) from Attendance with (nolock) where personalCode = ? and [Date] >= cast(GETDATE() as date))
                    begin
                        IF Exists (select 1 from Attendance where personalCode = ? and [Date] >= cast(GETDATE() as date) and exitTime is Null 
                                    and EntranceTime = (select MAX(atn.EntranceTime) from Attendance atn where atn.personalCode = personalCode and atn.[Date] >= cast(GETDATE() as date)) )
                            begin
                                if (select DATEDIFF(minute, atd.EntranceTime, CAST(getdate() as time)) from Attendance atd
                                    where atd.personalcode = ? and atd.[Date] = cast(getdate() as date) and atd.EntranceTime = (select MAX(atn.EntranceTime) from Attendance atn where atn.personalCode = atd.personalCode and atn.[Date] >= cast(GETDATE() as date))) > 1
                                begin
                                    update Attendance
                                    set exitTime = cast(getdate() as time)
                                    where personalCode = ? and [Date] >= cast(GETDATE() as date) and exitTime is null 
                                    and EntranceTime = (select MAX(atn.EntranceTime) from Attendance atn where atn.personalCode = personalCode and atn.[Date] >= cast(GETDATE() as date))
                                    set @Result = 1
                                end
                                else
                                begin
                                    set @Result = 3
                                end 
                            end
                        else
                            begin
                                if (select DATEDIFF(minute, atd.exitTime, CAST(getdate() as time)) from Attendance atd
                                    where atd.personalcode = ? and atd.[Date] = cast(getdate() as date) and atd.exitTime = (select MAX(atn.exitTime) from Attendance atn where atn.personalCode = atd.personalCode and atn.[Date] >= cast(GETDATE() as date))) > 1
                                begin
                                    insert into Attendance(personalCode, [Date], EntranceTime) values (?, cast(GETDATE() as date), cast(getdate() as time))
                                    set @Result = 1
                                end
                                else
                                begin
                                    set @Result = 3
                                end
                            end
                    end
                else
                    begin 
                        insert into Attendance(personalCode, [Date], EntranceTime) values (?, cast(GETDATE() as date), cast(getdate() as time))
                        set @Result = 1
                    end
                select @Result
            ''', int(userId), int(userId), int(userId), int(userId), int(userId), int(userId), int(userId)
        ).commit()
        rows = self.cursor.fetchall()
        self.cursor.close()
        for row in rows:
            print(row)
        # return row

    def CheckAttandance(self, userId):
        self.cursor.execute('Exec CheckAttendance @UserId=?', int(userId))
        data = self.cursor.fetchval()
        self.conn.commit()
        self.cursor.close()
        return data
    
    def ReadBulkQuery(self):
        self.cursor.execute("select * from Attendance")
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)

    
    def updateQuery(self):
        self.cursor.execute("UPDATE your_table SET column1 = ? WHERE column2 = ?", (new_value1, value2))
        self.conn.commit()

    
    def DeleteQuery(self):
        self.cursor.execute("DELETE FROM your_table WHERE column2 = ?", (value2,))
        self.conn.commit()

    def closeDBConnection(self): 
        try: 
            self.cursor.close() 
        except pyodbc.ProgrammingError: 
            pass 

if __name__=='__main__':
    sql = SqlQueries()
    # sql.CheckAttandance(813)
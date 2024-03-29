USE [humanResource]
GO
/****** Object:  StoredProcedure [dbo].[CheckAttendance]    Script Date: 12/12/2023 2:47:44 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
create or ALTER PROCEDURE [dbo].[CheckAttendance] 
	@UserId	 int = 0	
As
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;
	declare @Result  int = 0
	if exists (select (personalCode) from Attendance with (nolock) where personalCode = @UserId and [Date] >= cast(GETDATE() as date))
		begin
			IF Exists (select 1 from Attendance where personalCode = @UserId and [Date] >= cast(GETDATE() as date) and exitTime is Null 
						and EntranceTime = (select MAX(atn.EntranceTime) from Attendance atn where atn.personalCode = personalCode and atn.[Date] >= cast(GETDATE() as date)) )
				begin
					if (select DATEDIFF(minute, atd.EntranceTime, CAST(getdate() as time)) from Attendance atd
						where atd.personalcode = @UserId and atd.[Date] = cast(getdate() as date) and atd.EntranceTime = (select MAX(atn.EntranceTime) from Attendance atn where atn.personalCode = atd.personalCode and atn.[Date] >= cast(GETDATE() as date))) > 1
					begin
						--print('data exist')
						update Attendance
						set exitTime = cast(getdate() as time)
						where personalCode = @UserId and [Date] >= cast(GETDATE() as date) and exitTime is null 
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
					--print('print some data')
					if (select DATEDIFF(minute, atd.exitTime, CAST(getdate() as time)) from Attendance atd
						where atd.personalcode = @UserId and atd.[Date] = cast(getdate() as date) and atd.exitTime = (select MAX(atn.exitTime) from Attendance atn where atn.personalCode = atd.personalCode and atn.[Date] >= cast(GETDATE() as date))) > 1
					begin
						insert into Attendance(personalCode, [Date], EntranceTime) values (@UserId, cast(GETDATE() as date), cast(getdate() as time))
						set @Result = 1
					end
					else
					begin
						--print('whats going on')
						set @Result = 3
						--print('user is not allowed to pass the gate right now, please try after a minute.')
					end
				end
		end
	else
		begin 
			insert into Attendance(personalCode, [Date], EntranceTime) values (@UserId, cast(GETDATE() as date), cast(getdate() as time))
			set @Result = 1
		end
    -- Insert statements for procedure here
	select @Result
	return @Result;
END

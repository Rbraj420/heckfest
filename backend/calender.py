import calendar 
from datetime import datetime
yy= 2024
mm= 9
# print(calendar.month(yy,mm))
# print(datetime.now().date())
var= datetime.now().weekday()
cal= ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
print(cal[(var+1)%7])




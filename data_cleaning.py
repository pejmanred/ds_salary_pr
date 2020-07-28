import pandas as pd
from datetime import datetime,timedelta
import re
Salary_df=pd.read_csv('glassdoor_jobs.csv',index_col=False)
Salary_df=Salary_df.rename(columns={c: c.replace(' ','_') for c in Salary_df.columns}) #Replave the spaces in the columns name with "_"

Salary_df=Salary_df.query("Salary_Estimate != '-1' ") # Drop the rows with no values -- here is -1 --

Salary_df['Salary_Estimate'] = Salary_df['Salary_Estimate'].apply(lambda x: re.findall('\d+',x)) # Remove all but digits
Salary_df['Min_salary'] = Salary_df['Salary_Estimate'].apply(lambda x: int(x[0]))                   # Min salaries into a seprate columns
Salary_df['Max_salary'] = Salary_df['Salary_Estimate'].apply(lambda x: int(x[1]))       #Max salaries into a seprate columns
Salary_df['avg_salary']= Salary_df[["Min_salary","Max_salary"]].mean(axis=1)             

Salary_df['Company_Title']=Salary_df.Company_Name.apply(lambda x: x.split("\n")[0]) #remove the digits and cte from the company name column

Salary_df["State"]=Salary_df.Location.apply(lambda x: x.split(",")[1]) # keep the State and drop the cities
Salary_df["HeadQuarter_in_jobState"]=Salary_df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1) # find out if the job has in the company hradquarte

Salary_df["age"] = Salary_df.Founded.apply(lambda x: (datetime.now().year - x) if x > 1 else x ) #Age of the company

Salary_df['python']=Salary_df.Job_Description.apply(lambda x: 1 if "python" in x.lower() else 0) #How many "python" word is in Job description
Salary_df['r_sto']=Salary_df.Job_Description.apply(lambda x: 1 if "r studio" in x.lower() or "r-studio" in x.lower() else 0)
Salary_df['spark']=Salary_df.Job_Description.apply(lambda x: 1 if "spark" in x.lower()  else 0)
Salary_df['aws']=Salary_df.Job_Description.apply(lambda x: 1 if "aws" in x.lower()  else 0)
Salary_df['sql']=Salary_df.Job_Description.apply(lambda x: 1 if "sql" in x.lower()  else 0)

Salary_df_out= Salary_df.drop(['Unnamed:_0'],axis=1)
Salary_df_out.to_csv('Salary_cleaned.csv',index=False)

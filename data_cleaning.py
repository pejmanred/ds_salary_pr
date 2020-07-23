import pandas as pd
import re
Salary_df=pd.read_csv('glassdoor_jobs.csv',index_col=False)
Salary_df=Salary_df.rename(columns={c: c.replace(' ','_') for c in Salary_df.columns}) #Replave the spaces in the columns name with "_"

Salary_df=Salary_df.query("Salary_Estimate != '-1' ") # Drop the rows with no values -- here is -1 --

Salary_df['Salary_Estimate'] = Salary_df['Salary_Estimate'].apply(lambda x: re.findall('\d+',x))



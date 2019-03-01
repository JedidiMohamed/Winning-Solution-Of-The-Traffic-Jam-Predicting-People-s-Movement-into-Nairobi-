#EDA tool 
#Python tool Exploratory data analysis 

# characterization
class characterization:





	def __init__(self,user_csv_input,separation_input,output_csv_file,id_column):

           # input variable
		   user_input=user_csv_input
		   separation_input=separation_input
		   sfoo=output_csv_file
		   number_column_id=id_column

   		   import csv
		   from collections import defaultdict
		   import pandas as pd
		   import matplotlib
		   import numpy as np
		   import re     #this library is used so that I can use the "search" function
		   import os     #this is needed for using directory paths and manipulating them 
		   import sys
		   from datetime import datetime
		   from StringIO import StringIO
		   import math
           #Input data dataframe 
		   if ('.csv' in user_csv_input):
		   	data = pd.read_csv(user_input, sep=separation_input,na_filter=True ,skipinitialspace=False)
		   	datablink = pd.read_csv(user_input, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True,na_values='nan')
		   else:
		   	data = user_csv_input
		   	datablink = user_csv_input.dropna(how='any') 
		   

		   
		   a=list(data)
		   
		       
####################################################################################################
###################################### Type of every column ########################################
####################################################################################################
            
           # Vector  For different type
		   heuristics = (lambda value: datetime.strptime(value, "%m/%d/%y %I:%M"),int, float)
		   def convert(value):
					   	for type in heuristics:
					   		try:
					   			return type(value)
					   		except ValueError:
					   			continue
					   	return value


		   # List that containt all type of the data column
		   typeof= [] 
		   opl=0
		   for value in a:
		        
		        ii=0
		        jj=0
		        while( (ii<1)):
		           if (datablink.reset_index()[value][jj] !=  ''): 
		             check_type=datablink.reset_index()[value][jj]
		             
		             ii=ii+1
		             jj=jj+1
		           else:
		             jj=jj+1
		       
		   	converted_value = convert(str(check_type))
		        if(isinstance(converted_value, datetime)):
		   		typeof.append('Date & Time')
		        elif (data.dtypes[opl]=='object'):
		                typeof.append('String')
		   	elif(isinstance(converted_value, int)):
		   		typeof.append('Integer')
		   	elif(isinstance(converted_value, float)):
		   		typeof.append('Float')
		   	
		        opl=opl+1







		   
####################################################################################################
########################################## Count of element ########################################
####################################################################################################
           
           # Header data 
		   data_header=data.dtypes
		   data_matrix=data.as_matrix(columns=None)
		   length_data_matrix=len(data_matrix[0])

		   count= []
		   for countit in range(0,length_data_matrix):
		   	Count_val=pd.value_counts(pd.Series(data_matrix[:,countit]))
		   	counter=Count_val.count(level=None)
		   	count.append(counter)


####################################################################################################
########################################## Missing variable ########################################
####################################################################################################
			# Sum of the NAN value in the data
		   if_null=pd.isnull(data).sum()
		   tab= []
		   is_nullnuber=pd.isnull(data).sum().sort_values(ascending=False) 
		   for value in a:
		   	if(if_null[value]==0):
		   		tab.append('No')
		   	else:
		   		is_nullnuberv=is_nullnuber[value]
		   		lmp='Yes ('+str(is_nullnuberv)+')'
		   		tab.append(lmp)

		   typeofarray=np.array(typeof)
		   tabarray=np.array(tab)


####################################################################################################
########################################## Examples ################################################
####################################################################################################


		   example= [] 
		   ex_counter=0
		   for value in a:
		        countde=(pd.value_counts(pd.Series(data_matrix[:,ex_counter])).count(level=None))


		   	i=0
		   	var1=''
		        jl=0
		   	while((jl<data[value].size) & (i<6)):
		                
		                if (i==countde):
		                        if(typeof[ex_counter]=='Integer'):
		                            ss=str(int(float(data[data[value].notnull()][value].iloc[jl])))
		   			    var1 = var1 + ss
		   			    i=i+1
		                            break
		                        else:
		                            ss=str(data[data[value].notnull()][value].iloc[jl])
		   			    var1 = var1 + ss
		   			    i=i+1
		                            break
		                            
		                elif(i==0):
		                  if(typeof[ex_counter]=='Integer'):
		                        ss=str(int(float(data[data[value].notnull()][value].iloc[jl])))+'\n'                        
		   	                var1 = var1 + ss
		                        i=i+1
		                        ll=ss
		                  else:
		                        ss=str(data[data[value].notnull()][value].iloc[jl])+'\n'                        
		   	                var1 = var1 + ss
		                        i=i+1
		                        ll=ss
		   		elif (i<5):
		                     if(typeof[ex_counter]=='Integer'):
		   			ss=str(int(float(data[data[value].notnull()][value].iloc[jl])))+'\n'
		                        if(ll!=ss): 
		   			   var1 = var1 + ss
		   			   ll=ss
		                           i=i+1
		                     else:
		                        ss=str(data[data[value].notnull()][value].iloc[jl])+'\n'
		                        if(ll!=ss): 
		   			   var1 = var1 + ss
		   			   ll=ss
		                           i=i+1
		   		else:
		                   if(typeof[ex_counter]=='Integer'):
		   			ss=str(int(float(data[data[value].notnull()][value].iloc[jl])))
		   			var1 = var1 + ss
		   			i=i+1
		                   else:
		                        ss=str(data[data[value].notnull()][value].iloc[jl])
		   			var1 = var1 + ss
		   			i=i+1
		                jl=jl+1
		   	example.append(var1)
		        ex_counter=ex_counter+1
		   
####################################################################################################
########################################## description #############################################
####################################################################################################

		   def Change(v):
		   	l=''
		   	i=0
		   	for value in v:
		   		if(i>0):
		   			if(value=='_'):
		   				l=l+' '
		   			else: l=l+value
		   		i=i+1
		   	# convert upper case to lower case 
		   	l=l.lower()
		   	l=v[0]+l
		   	return l
		   	#list containt the description of every column 
		   des=[]
		   for value in a:
		   	des.append(Change(value))




		   x=[]
		   x=number_column_id.split(",")


####################################################################################################
########################################## Max  ####################################################
####################################################################################################

			#list that containt the max value of every column 
		   maxa=[]
		   for countss in range(0,length_data_matrix):
		   	# check the type before finding the max value 
		       if(((typeof[countss]=='Integer') or (typeof[countss]=='Float') )and ((a[countss] not in x))):
		           if(typeof[countss]=='Integer'):
		              maxa.append(int(float(data.iloc[:, countss].dropna().max())))
		           else:
		              maxa.append(data.iloc[:, countss].dropna().max())            
		       else:
		           maxa.append('')
		   
####################################################################################################
########################################## Min  ####################################################
####################################################################################################

			#list that containt the min value of every column 

		   Mina=[]
		   for countss in range(0,length_data_matrix):
		   	# check the type before finding the min value 
		       if(((typeof[countss]=='Integer') or (typeof[countss]=='Float') )and ((a[countss] not in x))):
		           if(typeof[countss]=='Integer'):
		              Mina.append(int(float(data.iloc[:, countss].dropna().min())))
		           else:
		              Mina.append(data.iloc[:, countss].dropna().min())

		       else:
		           Mina.append('')
####################################################################################################
########################################## Mean  ###################################################
####################################################################################################

			#list that containt the Mean value of every column  

		   Mean=[]
		   for countss in range(0,length_data_matrix):
		   	# check the type before finding the Mean value 
		       if(((typeof[countss]=='Integer') or (typeof[countss]=='Float') )and ((a[countss] not in x))):
		           if(typeof[countss]=='Integer'):
		              Mean.append(int(float(data.iloc[:, countss].dropna().mean())))
		           else:
		              Mean.append(data.iloc[:, countss].dropna().mean())

		       else:
		           Mean.append('')




####################################################################################################
########################################## Export the csv file  ####################################
####################################################################################################

		   d = pd.DataFrame({'Column name':data_header.axes[0],'Type':typeofarray,'Missing Value':tabarray,'Description':des,'Example':example,'Number Count':count,'Max':maxa,'Min':Mina,'Mean':Mean})
		   # reorganize the column  
		   dff = d.reindex_axis(['Column name','Type','Missing Value','Description','Number Count','Max','Min','Mean','Example'], axis=1)


		   try:
		   	os.remove(sfoo)
		   except OSError:
		   	pass
		   dff.to_csv(sfoo,encoding='utf-8')
		   print(" The Output file is in your rep with "+sfoo+" as a name !" )



# Mergin 2 csv file depending on one or several column 
class merging_file:


	def __init__(self,user_csv_file_1,separation_input_file_1,user_csv_file_2,separation_input_file_2,column_where_to_want,type_of_merge,output_csv_file):

			#import python module
			import csv
			from collections import defaultdict
			import pandas as pd
			import matplotlib
			import numpy as np
			import re     #this library is used so that I can use the "search" function
			import os     #this is needed for using directory paths and manipulating them 
			import sys
			from datetime import datetime
			from StringIO import StringIO
			import math

			#Input variable 
			user_input=user_csv_file_1
			separation_input=separation_input_file_1
			user_input1=user_csv_file_2
			separation_input1=separation_input_file_2
			number_column_merge=column_where_to_want
			type_merge=type_of_merge
			final_output=output_csv_file

			x=[]
			x=number_column_merge.split(",")
			if ('.csv' in user_input):
				data = pd.read_csv(user_input, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
			else:
				data = user_input
			if ('.csv' in user_input1):
				data1 = pd.read_csv(user_input1, sep=separation_input1,keep_default_na=False,na_filter=True,skip_blank_lines=True)
			else: 
				data1=user_input1

			result = pd.merge(data,data1,on=x,how=type_merge)
			try:
			  	os.remove(final_output)
			except OSError:
			  	pass
			result.to_csv(final_output, index=False)

			print("The Output file is in your rep with "+final_output+" as a name !" )

class Constraint_Validation:
	class fdv_verification:

		def __init__(self, input_csv_file,separation_input,key_column):
					import pandas as pd
					x=key_column.split(",")
					# Import retail sales data from an Excel Workbook into a data frame
					if ('.csv' in input_csv_file):
						df = pd.read_csv(input_csv_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
					else:
						df = input_csv_file

					# Let's add a new boolean column to our dataframe that will identify a duplicated order line item (False=Not a duplicate; True=Duplicate)
					df['is_duplicated'] = df.duplicated(x)

					# We are only interested in the duplicated order line items, so let's get those now
					df_nodup = df.loc[df['is_duplicated'] == True]
					print(df_nodup[x])

# Constraint validation 
	class validate_2_column:


		def __init__(self,user_input,separation_input,firstcolumn,secondcolumn,connectiontype):
		   

			#import python module
		   import csv
		   from collections import defaultdict
		   import pandas as pd
		   import matplotlib
		   import numpy as np
		   import re     #this library is used so that I can use the "search" function
		   import os     #this is needed for using directory paths and manipulating them 
		   import sys
		   from datetime import datetime
		   from StringIO import StringIO
		   import math
		   from functools import reduce

		   #input data csv file or dataframe
		   if ('.csv' in user_input):
		   			data = pd.read_csv(user_input, sep=separation_input,na_filter=True ,skipinitialspace=False)
		   else:
		   	        data = user_input

		   

		   # list of input data
		   a=list(data)
		   B=False
		   if (connectiontype=='1-1'):
		       #groupedby the secondcolumn depending on the first one
		       g = data[[firstcolumn, secondcolumn]].groupby(firstcolumn)
		       counts = g.transform(lambda x: len(x.unique()))
		       trueor=(counts == 1).all(axis=1)
		       #Convert data to matrix 
		       trimatrix=trueor.as_matrix(columns=None)
		       counto=counts[secondcolumn].size
		       if (False in trimatrix):
		         B=True
		         print('It is not a 1-1 connection there is problem in :')
		         for i in range(0,counto-1):
		                if (counts[secondcolumn][i]>1):
		                       print(data[firstcolumn][i])
		       #Check depending the second second column  
		       g = data[[firstcolumn, secondcolumn]].groupby(secondcolumn)
		       counts = g.transform(lambda x: len(x.unique()))
		       trueor=(counts == 1).all(axis=1)
		       trimatrix=trueor.as_matrix(columns=None)
		       counto=counts[firstcolumn].size
		       if (False in trimatrix):
		         B=True
		         print('there is problem in :')
		         for i in range(0,counto-1):
		                if (counts[firstcolumn][i]>1):
		                       print(data[secondcolumn][i])
		                       print('In row number '+str(i))
		       
		       if (not(B)):
		             print ('You have a 1-1 relation ') 

		   elif (connectiontype=='n-1'):

		       g = data[[firstcolumn, secondcolumn]].groupby(firstcolumn)
		       counts = g.transform(lambda x: len(x.unique()))
		       trueor=(counts == 1).all(axis=1)
		       trimatrix=trueor.as_matrix(columns=None)
		       counto=counts[secondcolumn].size
		       if (False in trimatrix):
		         B=True
		         print('It is not a n-1 connection there is problem in :')
		         for i in range(0,counto-1):
		                if (counts[secondcolumn][i]>1):
		                       print(data[firstcolumn][i])
		                       print('In row number '+str(i))

		       if (not(B)):
		             print ('You have a n-1 relation between '+firstcolumn+' and '+secondcolumn)


		   elif (connectiontype=='1-n'):

		       g = data[[firstcolumn, secondcolumn]].groupby(secondcolumn)
		       counts = g.transform(lambda x: len(x.unique()))
		       trueor=(counts == 1).all(axis=1)
		       trimatrix=trueor.as_matrix(columns=None)
		       counto=counts[firstcolumn].size
		       if (False in trimatrix):
		         B=True
		         print('It is not a 1-n connection there is problem in :')
		         for i in range(0,counto-1):
		                if (counts[firstcolumn][i]>1):
		                       print(data[secondcolumn][i])
		                       print('In row number '+str(i+1))

		       if (not(B)):
		             print ('You have a 1-n relation between '+firstcolumn+' and '+secondcolumn) 

		   else:
		       print( ' Couldn`t run . You typed an incorrect relation ,\n please choose between `1-1` or `1-n` or `n-1` relation ')
#=================================================================================================================		    
		                  
		                      
#Constraint validation on the holl 
	class validate_relationship_all_input:					             
			def __init__(self,user_input,separation_input,output_csv_file):

			   sfoo=output_csv_file

			   import csv
			   from collections import defaultdict
			   import pandas as pd
			   import matplotlib
			   import numpy as np
			   import re     #this library is used so that I can use the "search" function
			   import os     #this is needed for using directory paths and manipulating them 
			   import sys
			   from datetime import datetime
			   from StringIO import StringIO
			   import math
			   from functools import reduce

			   #input data (csv or dataframe) into pandas dataframes
			   if ('.csv' in user_input):
			   			data = pd.read_csv(user_input, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
			   else:
			   			data = user_input

			   # put all the header in a list 
			   all_header_list=list(data)
			   cout_header=list(data)
			   del all_header_list[-1]
			   Type_of_relation=[]
			   column_relation=[]

			   for valuei in all_header_list:
			     del cout_header[0]
			     for valuej in cout_header:
			        column_relation.append(valuei +' and '+valuej)
			        myFirstVar = True

			        myOtherVar = True

			        # Check the 1-n relation 
			        g = data[[valuei, valuej]].groupby(valuei)
			        counts = g.transform(lambda x: len(x.unique()))
			        trueor=(counts == 1).all(axis=1)
			        trimatrix=trueor.as_matrix(columns=None)
			        counto=counts[valuej].size
			        if (False in trimatrix):
			            myOtherVar=False

			         
			     	#Check the n-1 relation 
			        g = data[[valuei, valuej]].groupby(valuej)
			        counts = g.transform(lambda x: len(x.unique()))
			        trueor=(counts == 1).all(axis=1)
			        trimatrix=trueor.as_matrix(columns=None)
			        counto=counts[valuei].size
			        if (False in trimatrix):
			            myFirstVar=False

			        

			        if (myFirstVar and myOtherVar):
			            Type_of_relation.append('1-1')

			        elif (myFirstVar):
			         
			            Type_of_relation.append('1-n')
			        elif (myOtherVar):
			         
			            Type_of_relation.append('n-1')


			        else:
			         
			            Type_of_relation.append('n-n')


			   # Output file that contain all the relation between all the column 
			   d = pd.DataFrame({'Relation between':column_relation,'Relation Type':Type_of_relation})     

			   dff = d.reindex_axis(['Relation between','Relation Type'], axis=1)

			   try:
			   	os.remove(sfoo)
			   except OSError:
			   	pass
			   dff.to_csv(sfoo,encoding='utf-8')
			   print("The Output file is in your rep with "+sfoo+" as a name !" )


class Data_visualization:
# Chart that show the correlation between indicated features 
	class Correlation_between_features:

		def __init__(self, input_file,separation_input,feature_correlation_between_them):

			# Foundational modules
			            import pandas as pd # Version 0.19.2
			            import numpy as np # Version 1.12.1

						# Visualization modules
			            import matplotlib.pyplot as plt
			            import seaborn as sns
			            sns.set_style('white')
			            plt.rc('axes', titlesize=10)
			            #input data (csv or dataframe) into pandas dataframes
			            if ('.csv' in input_file):
			            		data = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
			            else:
			            		data = input_file


			            number_column_id=feature_correlation_between_them
			            plt.figure(figsize=(13,7))
			            x=[]
			            x=number_column_id.split(",")
			            data=data[x]
			            for feature in data.columns: # Loop through all columns in the dataframe
			                if data[feature].dtype == 'object': # Only apply for columns with categorical strings
			                   data[feature] = pd.Categorical(data[feature]).codes # Replace strings with an integer
			            corr = data.iloc[:, 1:].corr()
						# plot the heatmap correlation 
			            sns.heatmap(corr,cmap='RdBu', square=True, linewidth=0.6,annot=True, fmt='.2f')
			            plt.title(r"Feature Corrolation")

			            plt.show()

# Chart that show the Distribution between indicated 		
	class Distribution_Subset_Variable:

 		   	def __init__(self, input_file,separation_input,x_first_numeric_var,y_Second_numeric_var,Categorical_var,type_of_input):
						        

						            print('Loading ...')

						            

						# Foundational modules
						            import pandas as pd # Version 0.19.2
						            import numpy as np # Version 1.12.1

						# Visualization modules
						            import matplotlib.pyplot as plt
						            import seaborn as sns
						            sns.set_style('white');
						            plt.rc('axes', titlesize=10);


						            if ('.csv' in input_file):
						            			data = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True);
						            else :
						            			data = input_file
						            # Show Plot 
						            bplot2 = sns.jointplot(x=x_first_numeric_var, y=y_Second_numeric_var, data=data, size=10);
						            sns.lmplot(x=x_first_numeric_var, y=y_Second_numeric_var, hue=Categorical_var, data=data,palette="Set1",size=10)
						            plt.show()
						            
# Feature imporatances with 
	class Feature_importances_XGboost:

		def __init__(self, input_file,separation_input,target_variable):

									# Importing python module 
									import pandas as pd
									import matplotlib.pyplot as plt
									# load data
									if ('.csv' in input_file):
										df_raw = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
									else :
										df_raw = input_file

									
									for feature in df_raw.columns: # Loop through all columns in the dataframe
									    if df_raw[feature].dtype == 'object': # Only apply for columns with categorical strings
									        df_raw[feature] = pd.Categorical(df_raw[feature]).codes # Replace strings with an integer

									import xgboost as xgb
									# Xgboost Hyper parameters
									xgb_params = {
									    'eta': 0.5,
									    'max_depth': 15,
									    'subsample': 0.9,
									    'colsample_bytree': 0.9,
									    'objective': 'reg:linear',
									    'silent': 10,
									    'seed' : 10
									}
									# Traget Column 
									train_y = df_raw[target_variable].values
									# Training set 
									train_df = df_raw.drop([target_variable], axis=1)
									dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
									model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

									# plot the important features #
									fig, ax = plt.subplots(figsize=(10,10))
									xgb.plot_importance(model, height=0.5, ax=ax)
									plt.show()
		
# Bar plot between numerical and categorical variable 
	class  Bar_chart:

		def __init__(self, input_file,separation_input,y_numeric_var,x_categirical_var):


			# Importing Python modules 
		   import csv
		   from collections import defaultdict
		   import pandas as pd
		   import matplotlib
		   import numpy as np
		   import re     #this library is used so that I can use the "search" function
		   import os     #this is needed for using directory paths and manipulating them 
		   import sys
		   from datetime import datetime
		   from StringIO import StringIO
		   import math
		   import matplotlib.pyplot as plt


		   import numpy as np
		   import pandas as pd

		   # Input Variable 
		   user_input=input_file
		   separation_input=separation_input
		   Y_numeric=y_numeric_var
		   x_String=x_categirical_var
   			

		   #input data (csv or dataframe) into pandas dataframes

		   if ('.csv' in input_file):
		   		data = pd.read_csv(user_input, sep=separation_input,na_filter=True ,skipinitialspace=False)
		   		datablink = pd.read_csv(user_input, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True,na_values='nan')
		   else:
		   		data = user_input
		   		datablink = user_input

		   		# Header input data
		   a=list(data)
		   
		   ll=data.groupby(x_String).sum()[Y_numeric].sort_values(ascending=True)
		   plt.figure(figsize=(13,7))
		   # Extract influence['Change']: change
		   d = pd.DataFrame({'type':ll.axes[0],'Relation Type':ll})     
		   change = d['Relation Type']

		   #Description of your features
		   def Change(v):
		       l=''
		       i=0
		       for value in v:
		    
		           if(i>0):
		               if(value=='_'):
		                   l=l+' '
		               else: l=l+value
		           i=i+1
		       l=l.lower()
		       l=v[0]+l
		       return l           


		   des=[]
		   for value in a:
		       des.append(Change(value))

		    
		   # Make bar plot of change: ax
		   ax = change.plot(kind='bar')
		   y_name=des[data.columns.get_loc(Y_numeric)]
		   x_name=des[data.columns.get_loc(x_String)]
		   # Customize the plot to improve readability
		   ax.set_ylabel(y_name)
		   ax.set_title(y_name + ' in fonction of '+x_name)
		   ax.set_xticklabels(d['type'])
		   ax.set_xlabel(x_name)

		   # Display the plot
		   plt.show()





	class time_series_plot_with_2nd_categorical_var:
			def __init__(self, input_file,separation_input,date_column,date_format,date_range,sum_by,numerical_y_variable,Categorical_var,reassemble_type):


											import pandas as pd
											import matplotlib.pyplot as plt
											from datetime import datetime
											if ('.csv' in input_file):
												df1 = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
											else:
												df1 = input_file


											LOOP=df1[Categorical_var].unique()
											df1[date_column] = pd.to_datetime(df1[date_column], format=date_format)		
											frames = []
											j=0
											for i in LOOP:
											    
											    df = df1[df1[Categorical_var] == i]
											    df = df[[date_column,numerical_y_variable]]
											    df.index = df[date_column]
											    del df[date_column]
											    if (reassemble_type=='mean'):
											    	poi=df[date_range].resample(sum_by).mean()
											    elif (reassemble_type=='count'):
											    	poi=df[date_range].resample(sum_by).count()
											    else :
											    	poi=df[date_range].resample(sum_by).sum()
											    interpolated = poi.interpolate(method='spline', order=2)

											    frames.append(interpolated)
											    j=j+1



											df1 = df1[[date_column,numerical_y_variable]]


											df1.index = df1[date_column]
											del df1[date_column]

											if (reassemble_type=='mean'):	
												poi=df1[date_range].resample(sum_by).mean()
											elif (reassemble_type=='count'):
												poi=df1[date_range].resample(sum_by).count()
											else :
												poi=df[date_range].resample(sum_by).sum()
											#plt.plot(poi,label='QUANTITY')
											#plt.show()

											#upsampled = df1.resample('D')
											interpolated = poi.interpolate(method='spline', order=2)

											plt.figure(figsize=(16,14))
											ax = plt.subplot(2, 1, 1)

											plt.plot(interpolated, label=numerical_y_variable)
											plt.legend(shadow=True, fancybox=True)

											plt.figure(figsize=(16,14))

											ax = plt.subplot(2, 1, 1)
											for i in range(0,j):
											    plt.plot(frames[i], label=LOOP[i])
											plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
											           ncol=2, shadow=True, title=Categorical_var, fancybox=True)
											ax.get_legend().get_title().set_color("blue")
											plt.title('distribution')
											plt.show()

	class time_series_plot:
			def __init__(self, input_file,separation_input,date_column,date_format,date_range,sum_by,numerical_y_variable,reassemble_type):


											import pandas as pd
											import matplotlib.pyplot as plt
											from datetime import datetime
											if ('.csv' in input_file):
												df1 = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
											else:
												df1 = input_file


											df1[date_column] = pd.to_datetime(df1[date_column], format=date_format)		
											frames = []
											j=0




											df1 = df1[[date_column,numerical_y_variable]]


											df1.index = df1[date_column]
											del df1[date_column]

											if (reassemble_type=='mean'):	
												poi=df1[date_range].resample(sum_by).mean()
											elif (reassemble_type=='count'):
												poi=df1[date_range].resample(sum_by).count()
											else :
												poi=df1[date_range].resample(sum_by).sum()
											#plt.plot(poi,label='QUANTITY')
											#plt.show()

											#upsampled = df1.resample('D')
											interpolated = poi.interpolate(method='spline', order=2)

											plt.figure(figsize=(16,14))
											ax = plt.subplot(2, 1, 1)

											plt.plot(interpolated, label=numerical_y_variable)
											plt.legend(shadow=True, fancybox=True)

											ax.get_legend().get_title().set_color("blue")
											plt.title('distribution')
											plt.show()


	class box_plot_with_2_categorical_value:
			def __init__(self, input_file,separation_input,y_numerical_value,variable_to_map,x_categorical_value):
				import pandas as pd
				import matplotlib.pyplot as plt
				plt.rcParams['figure.figsize'] = [16, 10]
				import seaborn as sns
				if ('.csv' in input_file):
					df = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
				else:
					df = input_file

				flierprops = dict(markerfacecolor='0.66', markersize=5,linestyle='none')
				sns.boxplot(x=x_categorical_value, y=y_numerical_value, hue=variable_to_map, data=df,flierprops=flierprops,showfliers=False)
				plt.show()

	class box_plot:
			def __init__(self, input_file,separation_input,y_numerical_value,x_categorical_value):
				import pandas as pd
				import matplotlib.pyplot as plt
				plt.rcParams['figure.figsize'] = [16, 10]
				import seaborn as sns
				if ('.csv' in input_file):
					df = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
				else:
					df = input_file

				flierprops = dict(markerfacecolor='0.66', markersize=5,linestyle='none')
				sns.boxplot(x=x_categorical_value, y=y_numerical_value, data=df,flierprops=flierprops,showfliers=False)
				plt.show()


	class bar_plot_with_bins_range:
			def __init__(self, input_file,separation_input,bins,x_numerical_value,y_numerical_var):

										import numpy as np
										from itertools import cycle, islice
										import pandas as pd
										import matplotlib.pyplot as plt
										plt.rcParams['figure.figsize'] = [16, 10]
										import seaborn as sns
										if ('.csv' in input_file):
											df = pd.read_csv(input_file, sep=separation_input,keep_default_na=False,na_filter=True,skip_blank_lines=True)
										else:
											df = input_file
										rangeofx=x_numerical_value+' range'
										df[rangeofx] = pd.qcut(df[x_numerical_value],bins)
										var = df.groupby(rangeofx)[y_numerical_var].sum()
										width=0.8
										fig, ax = plt.subplots()
										x = [{i:np.random.randint(1,5)} for i in range(10)]
										my_colors = [(x/100.0, x/100.0, 0.75) for x in range(len(var))] # <-- Quick gradient example along the Red/Green dimensions.

										var.plot(kind='bar',figsize=(16,8),alpha=0.7,legend=True,rot=0, width=0.975,ax=ax,color=my_colors,stacked=True)
										plt.show()



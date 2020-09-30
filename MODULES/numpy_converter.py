import pandas as pd
import numpy as np


def process(file):
	print('db1 : '  + str(file))

	try:
		data=pd.read_csv(file)

		fullList = []
		lblList = []

		unq=data.ID.unique()
		print ("Unique Labels found :  "+ str(unq))
		for u_lbl in unq:
			data_no_label=data[data.ID==u_lbl]
			
			trialUnq = data_no_label.Name.unique()
			for trial in trialUnq:

				d_trial=data_no_label[data_no_label.Name==trial]

				res=d_trial.drop(columns=["Name","Sensor","ID"])
				#Feature data
				# print("Exercise# : " + str(trial))
				# print("Label# : " + str(u_lbl))
				# print("feature shape: " + str(res.shape))
				fullList.append(res.to_numpy())
				
				#Label data
				lbl = data_no_label['ID'].values[0]
				lblSubList = []
				lblSubList.append(str(lbl))
				# print ("Cur Label : " + str(lbl))
				if(lblList.count(lbl) == 0):
					lblList.append(lblSubList)
			
		npa = np.array(fullList)
		#print (str(npa))
		print ("\nFinal Feature shape : " + str(npa.shape))

		npLbl = np.array(lblList)
		#print (str(npLbl))
		print ("Label shape : " + str(npLbl.shape))

		return npa, npLbl

	except:
		print ("File read exception..")
		return (0,0)


def getGT(file):
	print('db2 : '  + str(file))

	try:
		data_gt=pd.read_csv(file)

		lblList = []

		unq=data_gt.ID.unique()
		print ("Unique Labels found :  "+ str(unq))
		for u_lbl in unq:
			data_no_label=data_gt[data_gt.ID==u_lbl]
			
			trialUnq = data_no_label.Name.unique()
			for trial in trialUnq:
				#Label data
				lbl = data_no_label['ID'].values[0]
				lblSubList = []
				lblSubList.append(str(lbl))
				# print ("Cur Label : " + str(lbl))
				if(lblList.count(lbl) == 0):
					lblList.append(lblSubList)
			

		npLbl = np.array(lblList)
		#print (str(npLbl))
		print ("Label shape : " + str(npLbl.shape))

		return npLbl

	except:
		print ("File read exception..")
		return 0
	

if __name__ == '__main__':
	main()
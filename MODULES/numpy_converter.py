import pandas as pd
import numpy as np


def process(file):
	try:
		data=pd.read_csv(file)
	except:
		print ("File read exception..")
	
	unq=data.Name.unique()

	fullList = []
	lblList = []

	for u in unq:
		d=data[data.Name==u]
		res=d.drop(columns=["Name","Sensor","ID"])
		#Feature data
		fullList.append(res.to_numpy())
		#Label data
		lbl = d['ID'].values[0]
		lblSubList = []
		lblSubList.append(str(lbl))
		if(lblList.count(lbl) == 0):
			lblList.append(lblSubList)

	npa = np.array(fullList)
	#print (str(npa))
	print ("Feature shape : " + str(npa.shape))

	npLbl = np.array(lblList)
	#print (str(npLbl))
	print ("Label shape : " + str(npLbl.shape))

	return npa, npLbl


if __name__ == '__main__':
	main()
import pandas as pd
import numpy as np


def process(file):
	data=pd.read_csv(file)
	unq=data.id.unique()

	fullList = []
	lblList = []

	for u in unq:
		d=data[data.id==u]
		res=d.drop(columns=["name","sensor","id"])
		fullList.append(res.to_numpy())
		if(lblList.count(u) == 0):
			lblList.append(u)

	npa = np.array(fullList)
	#print str(npa)
	print ("Feature shape : " + str(npa.shape))

	npLbl = np.array(lblList)
	#print str(npLbl)
	print ("Label shape : " + str(npLbl.shape))
	#print (lblNumData.shape)

	return npa, npLbl


if __name__ == '__main__':
	FILE_NAME=["../MEMS_ML_data - ID_1.csv"]
	for file in FILE_NAME:
		process(file)
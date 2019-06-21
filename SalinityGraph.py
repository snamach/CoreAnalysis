import sys
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import style
style.use("ggplot")

def reader(file):
	f=open(file)
	xs=f.readlines()
	xs=[float(i[:-2]) for i in xs]
	return np.asarray(xs)

def get_colors(n):
	colors=[]

	for i in np.arange(0.,360.,360./n):
		hue=i/360.
		lightness=(50+np.random.rand()*10)/100.
		saturation=(90+np.random.rand()*10)/100.
		colors.append(colorsys.hls_to_rgb(hue,lightness,saturation))
	
	return colors

def exp(x,a,b,c):
	return a*np.exp(-b*x)+c

def cubic(x,a,b,c,d):
	return a*x**3+b*x**2+c*x+d

def sigmoid(x,a,b):
	return 1/(1+np.exp(a*(x+b)))

def regress(core,mode):
	Elevations=reader(core+"E.txt")
	Salinities=reader(core+"S.txt")

	if mode=="Exp":
		popt,pcov=curve_fit(exp,Elevations,Salinities)
		coeff=[]

		for i in popt:
			coeff.append(i)

		residuals=Salinities-exp(Elevations,*popt)

	if mode=="Cub":
		popt,pcov=curve_fit(cubic,Elevations,Salinities)
		coeff=[]

		for i in popt:
			coeff.append(i)

		residuals=Salinities-cubic(Elevations,*popt)
           
	if mode=="Sig":
		S_min=min(Salinities)
		S_max=max(Salinities)
		S_trans=(Salinities-S_min)/(S_max-S_min)
		popt,pcov=curve_fit(sigmoid,Elevations,S_trans)
		coeff=[]
        
		for i in popt:
			coeff.append(i)

		residuals=Salinities-((S_max-S_min)*sigmoid(Elevations,*popt)+S_min)

	ss_res=np.sum(residuals**2)
	ss_tot=np.sum((Salinities-np.mean(Salinities))**2)
	r2=1-(ss_res/ss_tot)
	
	return np.asarray(coeff),r2

def avgfit(corelist,mode):
	colors=get_colors(len(corelist))
	i=0
	fig,ax=plt.subplots()
	ax.set(xlabel="Elevation(m)",ylabel="Salinity(psu)",title="Salinity vs Elevation")
	ax.grid(b=True)
	Elevations=np.empty(0)
	Salinities=np.empty(0)

	for core in corelist:
		E=reader(core+"E.txt")
		Elevations=np.concatenate((Elevations,E))
		S=reader(core+"S.txt")
		Salinities=np.concatenate((Salinities,S))
		ax.plot(E,S,color=colors[i],marker="o",linestyle="None",label=core)
		i+=1

	E_min=min(Elevations)
	E_max=max(Elevations)
	x_vals=np.arange(start=E_min,stop=E_max,step=(E_max-E_min)/50)
    
	if mode=="Exp":
		avgcoeffs=np.asarray([0.,0.,0.])

	if mode=="Cub":
		avgcoeffs=np.asarray([0.,0.,0.,0.])

	if mode=="Sig":
		S_min=min(Salinities)
		S_max=max(Salinities)
		avgcoeffs=np.asarray([0.,0.])

	for core in corelist:
		avgcoeffs+=regress(core,mode)[0]

	avgcoeffs=avgcoeffs/len(corelist)

	if mode=="Exp":
		equation=str(avgcoeffs[0])+"e^"+"("+str(avgcoeffs[1]*-1)+"x"+")"+"+"+str(avgcoeffs[2])
		y_vals=exp(x_vals,avgcoeffs[0],avgcoeffs[1],avgcoeffs[2])
		residuals=Salinities-exp(Elevations,avgcoeffs[0],avgcoeffs[1],avgcoeffs[2])

	if mode=="Cub":
		equation=str(avgcoeffs[0])+"x^3"+"+"+str(avgcoeffs[1])+"x^2"+"+"+str(avgcoeffs[2])+"x"+"+"+str(avgcoeffs[3])
		y_vals=cubic(x_vals,avgcoeffs[0],avgcoeffs[1],avgcoeffs[2],avgcoeffs[3])
		residuals=Salinities-cubic(Elevations,avgcoeffs[0],avgcoeffs[1],avgcoeffs[2],avgcoeffs[3])

	if mode=="Sig":
		equation=str(S_max-S_min)+"/(1+e^("+str(avgcoeffs[0])+"*(x+"+str(avgcoeffs[1])+")))"+"+"+str(S_min)
		y_vals=(S_max-S_min)*sigmoid(x_vals,avgcoeffs[0],avgcoeffs[1])+S_min
		residuals=Salinities-((S_max-S_min)*sigmoid(Elevations,avgcoeffs[0],avgcoeffs[1])+S_min)

	ax.plot(x_vals,y_vals,"k-",label="Fitted Curve")
	ax.legend()
	fig.savefig("Avg"+mode+".png")

	ss_res=np.sum(residuals**2)
	ss_tot=np.sum((Salinities-np.mean(Salinities))**2)
	r2=1-(ss_res/ss_tot)
	outf=open("Avg"+mode+".txt","w+")
	outf.write("Total R-squared="+str(r2)+"\n")
	outf.write(equation+"\n")

	for core in corelist:
		E=reader(core+"E.txt")
		S=reader(core+"S.txt")
	
		if mode=="Exp":
			res=S-exp(E,avgcoeffs[0],avgcoeffs[1],avgcoeffs[2])

		if mode=="Cub":
			res=S-cubic(E,avgcoeffs[0],avgcoeffs[1],avgcoeffs[2],avgcoeffs[3])

		if mode=="Sig":
			res=S-((S_max-S_min)*sigmoid(E,avgcoeffs[0],avgcoeffs[1])+S_min)

		ss_res=np.sum(res**2)
		ss_tot=np.sum((S-np.mean(S))**2)
		r2=1-(ss_res/ss_tot)
		outf.write(core+" R-squared="+str(r2)+"\n")

def wavgfit(corelist,mode):
	colors=get_colors(len(corelist))
	i=0
	fig,ax=plt.subplots()
	ax.set(xlabel="Elevation(m)",ylabel="Salinity(psu)",title="Salinity vs Elevation")
	ax.grid(b=True)
	Elevations=np.empty(0)
	Salinities=np.empty(0)

	for core in corelist:
		E=reader(core+"E.txt")
		Elevations=np.concatenate((Elevations,E))
		S=reader(core+"S.txt")
		Salinities=np.concatenate((Salinities,S))
		ax.plot(E,S,color=colors[i],marker="o",linestyle="None",label=core)
		i+=1

	E_min=min(Elevations)
	E_max=max(Elevations)
	x_vals=np.arange(start=E_min,stop=E_max,step=(E_max-E_min)/50)
    
	if mode=="Exp":
		wavgcoeffs=np.asarray([0.,0.,0.])

	if mode=="Cub":
		wavgcoeffs=np.asarray([0.,0.,0.,0.])

	if mode=="Sig":
		S_min=min(Salinities)
		S_max=max(Salinities)
		wavgcoeffs=np.asarray([0.,0.])

	denom=0.

	for core in corelist:
		coeffs,wght=regress(core,mode)
		denom+=wght
		wavgcoeffs+=(wght*coeffs)

	wavgcoeffs=wavgcoeffs/denom

	if mode=="Exp":
		equation=str(wavgcoeffs[0])+"e^"+"("+str(wavgcoeffs[1]*-1)+"x"+")"+"+"+str(wavgcoeffs[2])
		y_vals=exp(x_vals,wavgcoeffs[0],wavgcoeffs[1],wavgcoeffs[2])
		residuals=Salinities-exp(Elevations,wavgcoeffs[0],wavgcoeffs[1],wavgcoeffs[2])

	if mode=="Cub":
		equation=str(wavgcoeffs[0])+"x^3"+"+"+str(wavgcoeffs[1])+"x^2"+"+"+str(wavgcoeffs[2])+"x"+"+"+str(wavgcoeffs[3])
		y_vals=cubic(x_vals,wavgcoeffs[0],wavgcoeffs[1],wavgcoeffs[2],wavgcoeffs[3])
		residuals=Salinities-cubic(Elevations,wavgcoeffs[0],wavgcoeffs[1],wavgcoeffs[2],wavgcoeffs[3])

	if mode=="Sig":
		equation=str(S_max-S_min)+"/(1+e^("+str(wavgcoeffs[0])+"*(x+"+str(wavgcoeffs[1])+")))"+"+"+str(S_min)
		y_vals=(S_max-S_min)*sigmoid(x_vals,wavgcoeffs[0],wavgcoeffs[1])+S_min
		residuals=Salinities-((S_max-S_min)*sigmoid(Elevations,wavgcoeffs[0],wavgcoeffs[1])+S_min)

	ax.plot(x_vals,y_vals,"k-",label="Fitted Curve")
	ax.legend()
	fig.savefig("wAvg"+mode+".png")

	ss_res=np.sum(residuals**2)
	ss_tot=np.sum((Salinities-np.mean(Salinities))**2)
	r2=1-(ss_res/ss_tot)
	outf=open("wAvg"+mode+".txt","w+")
	outf.write("Total R-squared="+str(r2)+"\n")
	outf.write(equation+"\n")

	for core in corelist:
		E=reader(core+"E.txt")
		S=reader(core+"S.txt")
	
		if mode=="Exp":
			res=S-exp(E,wavgcoeffs[0],wavgcoeffs[1],wavgcoeffs[2])

		if mode=="Cub":
			res=S-cubic(E,wavgcoeffs[0],wavgcoeffs[1],wavgcoeffs[2],wavgcoeffs[3])

		if mode=="Sig":
			res=S-((S_max-S_min)*sigmoid(E,wavgcoeffs[0],wavgcoeffs[1])+S_min)

		ss_res=np.sum(res**2)
		ss_tot=np.sum((S-np.mean(S))**2)
		r2=1-(ss_res/ss_tot)
		outf.write(core+" R-squared="+str(r2)+"\n")

def aggfit(corelist,mode):
	colors=get_colors(len(corelist))
	i=0
	fig,ax=plt.subplots()
	ax.set(xlabel="Elevation(m)",ylabel="Salinity(psu)",title="Salinity vs Elevation")
	ax.grid(b=True)
	Elevations=np.empty(0)
	Salinities=np.empty(0)

	for core in corelist:
		E=reader(core+"E.txt")
		Elevations=np.concatenate((Elevations,E))
		S=reader(core+"S.txt")
		Salinities=np.concatenate((Salinities,S))
		ax.plot(E,S,color=colors[i],marker="o",linestyle="None",label=core)
		i+=1

	E_min=min(Elevations)
	E_max=max(Elevations)
	x_vals=np.arange(start=E_min,stop=E_max,step=(E_max-E_min)/50)

	if mode=="Exp":
		popt,pcov=curve_fit(exp,Elevations,Salinities)
		coeff=[]

		for i in popt:
			coeff.append(i)

		equation=str(coeff[0])+"e^"+"("+str(coeff[1]*-1)+"x"+")"+"+"+str(coeff[2])
		y_vals=exp(x_vals,*popt)
		ax.plot(x_vals,y_vals,"k-",label="Fitted Curve")
		residuals=Salinities-exp(Elevations,*popt)

	if mode=="Cub":
		popt,pcov=curve_fit(cubic,Elevations,Salinities)
		coeff=[]

		for i in popt:
			coeff.append(i)

		equation=str(coeff[0])+"x^3"+"+"+str(coeff[1])+"x^2"+"+"+str(coeff[2])+"x"+"+"+str(coeff[3])
		y_vals=cubic(x_vals,*popt)
		ax.plot(x_vals,y_vals,"k-",label="Fitted Curve")
		residuals=Salinities-cubic(Elevations,*popt)
           
	if mode=="Sig":
		S_min=min(Salinities)
		S_max=max(Salinities)
		S_trans=(Salinities-S_min)/(S_max-S_min)
		popt,pcov=curve_fit(sigmoid,Elevations,S_trans)
		coeff=[]
        
		for i in popt:
			coeff.append(i)

		equation=str(S_max-S_min)+"/(1+e^("+str(coeff[0])+"*(x+"+str(coeff[1])+")))"+"+"+str(S_min)
		y_vals=(S_max-S_min)*sigmoid(x_vals,*popt)+S_min
		ax.plot(x_vals,y_vals,"k-",label="Fitted Curve")
		residuals=Salinities-((S_max-S_min)*sigmoid(Elevations,*popt)+S_min)

	ax.legend()
	fig.savefig("Agg"+mode+".png")

	ss_res=np.sum(residuals**2)
	ss_tot=np.sum((Salinities-np.mean(Salinities))**2)
	r2=1-(ss_res/ss_tot)

	outf=open("Agg"+mode+".txt","w+")
	outf.write("R-squared="+str(r2)+"\n")
	outf.write(equation+"\n")

	for core in corelist:
		E=reader(core+"E.txt")
		S=reader(core+"S.txt")
	
		if mode=="Exp":
			res=S-exp(E,coeff[0],coeff[1],coeff[2])

		if mode=="Cub":
			res=S-cubic(E,coeff[0],coeff[1],coeff[2],coeff[3])

		if mode=="Sig":
			res=S-((S_max-S_min)*sigmoid(E,coeff[0],coeff[1])+S_min)

		ss_res=np.sum(res**2)
		ss_tot=np.sum((S-np.mean(S))**2)
		r2=1-(ss_res/ss_tot)
		outf.write(core+" R-squared="+str(r2)+"\n")

args=sys.argv[1:]
reg_type=args[0]
fit=args[1]
corelist=args[2:]		

if fit=="avg":
	avgfit(corelist,reg_type)
	
if fit=="wavg":
	wavgfit(corelist,reg_type)

if fit=="agg":
	aggfit(corelist,reg_type)

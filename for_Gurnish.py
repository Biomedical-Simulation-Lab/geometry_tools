#add this line at the top:
import matplotlib.pyplot as plt 

#at line 255 of map_info.py add the following: NOTE this is untested, but this is the bones.

rho=1057
U_c = (m.centerlines.point_data['flowrate']/m.centerlines.point_data['CSA'])
m.centerlines.point_data['dP']=0.5*rho*(3/2*U_c)**2/133.322 #dP based on max centerline velocity in mmHg calculated in every branch

size = 10
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
plt.rc('legend', fontsize=size) #fontsize of the legend

#NOTE: this should follow the centerlines along the main branch but it is possible that the point numbering for the main branch is out of order (not what you would expect) and the displacement won't be correct and the dP's may not line up either- check this. If so I can probably figure something out when I get back if you can't get it to work out.

def line_cm(pts=np.flip(m.centerlines.points[m.centerlines.points['main_branch']==1],axis=0)): #centerline numbering goes from outlet to inlet? this is the line you may need to change
        seg_lens=np.zeros(len(pts))
        seg_lens[1:-1]=np.sqrt(np.sum(np.square(pts[0:-2,:]-pts[1:-1, :]),1)) #length of each segment
        disp=np.array([np.sum(seg_lens[0:ii]) for ii in range(len(seg_lens))]) #displacement in mm
        return disp/10 #displacement in cm
        
x = line_cm()
plt.figure(figsize=(7, 4))
dP_main = m.centerlines.point_data['dP'][m.centerlines.points['main_branch']==1] #only take the pressure drop calculated on the main branch
plt.plot(x, dP_main,color='b', label='$dP$', linewidth=0.5)
plt.xlabel('Axial Position (cm)', labelpad=-1)
plt.ylabel('Pressure Drop (mmHg)', labelpad=-4)
plt.savefig('Pressuredrop.png') #save figure
np.savez('Pressuredrop.npz', x=x, dP_main=dP_main) #save the data for later


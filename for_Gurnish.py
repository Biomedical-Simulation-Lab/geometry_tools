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

def line_cm(pts,axis=0):
        seg_lens=np.zeros(len(pts))
        seg_lens[1:-1]=np.sqrt(np.sum(np.square(pts[0:-2,:]-pts[1:-1, :]),1)) #length of each segment
        disp=np.array([np.sum(seg_lens[0:ii]) for ii in range(len(seg_lens))]) #displacement in mm
        return disp/10 #displacement in cm

#Order the points along the main branch - this will work for the unilateral case
unordered_points = m.centerlines.points[m.centerlines.point_data['main_branch']==1] #for bilateral case, use [m.centerlines.point_data['main_branch_l'] != 0] or [m.centerlines.point_data['main_branch_r'] != 0] as the index instead

# Set a seed point to start the ordering
# Find the index of the point with the highest Z value, which should be the SSS inlet, but check this!! Can also choose a different seed point (eg. the outlet
highest_z_index = np.argmax(unordered_points[:, 2])
seed_point = unordered_points[highest_z_index]
seed_index = highest_z_index

# Remove the seed point from the list of unordered points
remaining_points = np.delete(unordered_points, seed_index, axis=0)
ordered_points = [seed_point]

while remaining_points.shape[0] > 0:
	kdtree = cKDTree(remaining_points)
    dist, index = kdtree.query(ordered_points[-1])
    nearest_point = remaining_points[index]
    ordered_points.append(nearest_point)
    remaining_points = np.delete(remaining_points, index, axis=0)

ordered_points = np.array(ordered_points)
        
x = line_cm(ordered_points)
plt.figure(figsize=(7, 4))
dP_main = m.centerlines.point_data['dP'][m.centerlines.points['main_branch']==1] #only take the pressure drop calculated on the main branch
plt.plot(x, dP_main,color='b', label='$dP$', linewidth=0.5)
plt.xlabel('Axial Position (cm)', labelpad=-1)
plt.ylabel('Pressure Drop (mmHg)', labelpad=-4)
plt.savefig('Pressuredrop.png') #save figure
np.savez('Pressuredrop.npz', x=x, dP_main=dP_main) #save the data for later


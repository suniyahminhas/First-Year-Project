'''The main project python'''
USER="Suniyah_Minhas"
USER_ID="sppt67"

import numpy

'''Reference: Part of this was taken and adapted from ray_mini_solution provided by the department online'''

'''Ray_mini_solution'''
def refraction_2d (incident_rays, planar_surface):
    '''This is a function to determine the points of intersection and angle, to the vertical, of rays passing through a planar surface
	it is designed to raise an exception should any incoming rays exceed the critical angle upon intersection
	
    incident_rays consist of a 2d numpy array of floats, the ray is specified by the first index, the second index 
    specifies (in order) horizontal coordinate of ray origin, corresponding vertical coordinate (both in metres) and the clockwise
    angle (in radians), with respect to positive vertical direction, of incident ray
    
    The planar_surface is a 1d numpy array of floats specifying (in order) horizontal coordinate of one end, corresponding vertical coordinate,
    horizontal coordinate of other end, corresponding vertical coordinate (all in metres), refractive index on incident side, refractive index of 
    opposing side 
	
    The refracted_rays we return will be a 2d numpy array where first index will determine the ray and second index will specify (in order)
    the horizontal coordinate of point of intersection with corresponding vertical coordinate (in metres) and the clockwise angle with respect 
    to positive vertical direction of the refracted ray (in radians)'''
    
    # set up array to hold results
    refracted_rays = numpy.zeros(incident_rays.shape, dtype=float)
    # this will avoid a transpose when constructing the returned arraynumpy.zeros(incident_rays.shape, dtype=float) of results
    
    # calculate critical angle
    if planar_surface[5] >= planar_surface[4]:
        # total internal reflection is not possible - save computational time by not calculating the critical angle
        tir_possible = False
    else:
        tir_possible = True
        critical_angle = numpy.arcsin(planar_surface[5]/planar_surface[4])
    
    # calculate angles to surface normal of incident rays
    planar_angle = numpy.arctan((planar_surface[2]-planar_surface[0])/(planar_surface[3]-planar_surface[1]))
    incident_angles =  incident_rays[:,2] - (numpy.pi/2.0) - planar_angle # transform ray angles to be with respect to normal to surface
    
    # handle incident rays exceeding the critical angle
    if tir_possible and(abs(incident_angles) > critical_angle).any():
        raise Exception, "at least one incident ray exceeds the critical angle"
    
    # calculate gradients and intercepts of incoming rays and surface
    ray_gradients = numpy.tan((numpy.pi/2.0) - incident_rays[:,2])
    ray_intercepts = incident_rays[:,1] - (ray_gradients * incident_rays[:, 0])
    surface_gradient = (planar_surface[3]-planar_surface[1])/(planar_surface[2]-planar_surface[0])
    surface_intercept = planar_surface[1] - (surface_gradient * planar_surface[0])

    # calculate points of intersection of rays with surface...
    # horizontal
    refracted_rays[:,0] = (surface_intercept - ray_intercepts) / (ray_gradients - surface_gradient)
    # vertical
    refracted_rays[:,1] = (ray_gradients * refracted_rays[:,0]) + ray_intercepts
    
    # calculate directions of refracted rays
    refracted_angles = numpy.arcsin(planar_surface[4]*numpy.sin(incident_angles)/planar_surface[5]) # Snell's law
    refracted_rays[:,2] = refracted_angles + (numpy.pi/2.0) + planar_angle # transform output ray angles to be clockwise from vertical
      
    return refracted_rays

'''task 1'''
def refraction_2d_sph (incident_rays, spherical_surface):
    '''This is a function designed to determine the points of intersection and angle, to the vertical, of rays passing through a spherical surface
	which is symmetric about the x-axis
	
	incident_rays consist of a 2d numpy array of floats, the ray is specified by the first index, the second index 
        specifies (in order) horizontal coordinate of ray origin, corresponding vertical coordinate (both in metres) and the clockwise
        angle (in radians) with respect to positive vertical direction of incident ray
	
        spherical_surface is a 1d numpy array specifying horizontal coordinate of one end, corresponding vertical coordinate, 
	horizontal coordinate of other end, coresponding vertical coordinate, refractive index of incident side, refractive index of
	other side, radius of curvuture (which is positive if surface is convex, negative if concave)
	
	the refracted_rays we return will be a 2d numpy array where first index will determine the ray and second index will specify (in order)
        the horizontal coordinate of point of intersection with corresponding vertical coordinate (in metres) and the clockwise angle with respect 
        to positive vertical direction of the refracted ray (in radians)'''
    
    # set up array to hold results
    refracted_rays = numpy.zeros(incident_rays.shape, dtype=float)
    
    #  Find centre points of the spherical surface
    if spherical_surface[6]>0:
        spherical_centre=spherical_surface[0]+numpy.sqrt(((spherical_surface[6])**2)-((spherical_surface[1])**2)) 
		#if radius is postive (i.e. convex) then centre will be on the right of the end points
    if spherical_surface[6]<0:
        spherical_centre=spherical_surface[0]-numpy.sqrt(((spherical_surface[6])**2)-((spherical_surface[1])**2))
		#if radius is negative (i.e. concave) then centre will be on the left of the end points
	
	
    # calculate gradients and intercepts of incoming rays and surface
    ray_gradients = numpy.tan((numpy.pi/2.0) - incident_rays[:,2])
    ray_intercepts = incident_rays[:,1] - (ray_gradients * incident_rays[:,0])
    
    #a quadratic is then formed when combining equation of line and equation of the spherical surface
    #work out the values of the quadratic 
    a=1+(ray_gradients)**2
    b=(2*(ray_gradients)*(ray_intercepts))-(2*spherical_centre)
    c=((spherical_centre)**2)+((ray_intercepts)**2)-((spherical_surface[6])**2)
    
    #solve the quadratic to find x intercept
    if spherical_surface[6]>0:
        refracted_rays[:,0]=(-b-numpy.sqrt((b**2)-4*a*c))/(2*a)
		#if radius is postive (i.e. convex) then intersection will be on the left of end points
    if spherical_surface[6]<0:
        refracted_rays[:,0]=(-b+numpy.sqrt((b**2)-4*a*c))/(2*a)
		#if radius is negative (i.e. concave) then intersection will be on the right of end points
		
    #find y intercepts- by replacing x into equation of the ray line
    refracted_rays[:,1]=ray_gradients*refracted_rays[:,0]+ray_intercepts
    
    #find gradient and intercept of tangent to the surface at point of intersection
    normal_gradients=(refracted_rays[:,1])/(refracted_rays[:,0]-spherical_centre)
    tangent_gradients=-1/normal_gradients
    tangent_intercepts=refracted_rays[:,1]-(tangent_gradients*refracted_rays[:,0])
  
    # calculate directions of refracted rays (just as with planar surface but with the planar surface acting as the tangent of the surface at the intersection point
    planar_angle = numpy.arctan(1/tangent_gradients)
    incident_angles =  incident_rays[:,2] - (numpy.pi/2.0) - planar_angle # transform ray angles to be with respect to normal to surface
    refracted_angles = numpy.arcsin(spherical_surface[4]*numpy.sin(incident_angles)/spherical_surface[5]) # Snell's law
    refracted_rays[:,2] = refracted_angles + (numpy.pi/2.0) + planar_angle # transform output ray angles to be clockwise from vertical
    
	
    return refracted_rays

'''task 2'''
def refraction_2d_det (incident_rays, x_det):
    ''' This is a function to find point of intersection with a detector surface, which is a fixed
	line parllel to the y axis with the equation x=x_det (x_det is therefore simply given as a float value)
	
	incident_rays consist of a 2d numpy array of floats, the ray is specified by the first index, the second index 
    specifies (in order) horizontal coordinate of ray origin, corresponding vertical coordinate (both in metres) and the clockwise
    angle (in radians) with respect to positive vertical direction of incident ray
	
	the refracted_rays we return will be a 2d numpy array where first index will determine the ray and second index will specify (in order)
    the horizontal coordinate of point of intersection with corresponding vertical coordinate (in metres) and the clockwise angle with respect 
    to positive vertical direction of the refracted ray (in radians)'''
    
    # set up array to hold results
    refracted_rays = numpy.zeros(incident_rays.shape, dtype=float)
    # This will avoid a transpose when constructing the returned array of results
    
    # calculate gradients and intercepts of incoming rays 
    ray_gradients = numpy.tan((numpy.pi/2.0) - incident_rays[:,2])
    ray_intercepts = incident_rays[:,1] - (ray_gradients * incident_rays[:, 0])
    
	# calculate points of intersection of rays with surface
    # vertical- input x_det into equation of ray
    refracted_rays[:,1] = (ray_gradients * x_det) + ray_intercepts
	# horizontal intersection (will always be x_det
    refracted_rays[:,0] = x_det
    
	#angles do not need to be edited as simply given as zero (the ray will not travel further than the detector)
	
    return refracted_rays
    
'''task 3'''

def trace_2d (incident_rays, surface_list):
    '''Function to trace multiple rays through multiple surfaces, including planar surfaces, spherical surfaces and detectors
	
    incident_rays consist of a 2d numpy array of floats, the ray is specified by the first index, the second index 
    specifies (in order) horizontal coordinate of ray origin, corresponding vertical coordinate (both in metres) and the clockwise
    angle (in radians) with respect to positive vertical direction of incident ray
	
    surface_list is a list consisting of seperate lists where the first element is a string saying either 'PLA', 'SPH' or 'DET', 
    this therefore specifies what type of surface is being met, the second element is the corresponding numpy array (or float value) 
    so for PLA this would be planar_surface (as defined in refraction_2d), for SPH this would be spherical_surface (as defined in refraction_2d_sph)
    and for DET this would be x_det (as defined in refraction_2d_det)
	
    refracted ray paths is a 3 dimensional array, of the refracted_rays (from refraction_2d, refraction_2d_sph, refraction_2d_det) for each surface in turn,
    therefore the first index represents the surface and the next two represent the same things as refracted_rays (the return of refraction_2d or refraction_2d_sph
    or refraction_2d_det)'''
	
    # set up the shape of the refracted ray paths
    refracted_ray_paths = numpy.zeros([len(surface_list),incident_rays.shape[0], incident_rays.shape[1]])
	
	#we need to go through every value of the surface_list
    for i in range(len(surface_list)):
        if surface_list[i][0] == 'PLA':
            planar_surface=surface_list[i][1] #if dealing with 'PLA' then we need to cut out the planar_surface from surface_list
            refracted_ray_paths[i,:,:]=refraction_2d (incident_rays, planar_surface) #we input the 2d refracted_rays from refraction_2d into the corresponding section of the 3d refracted_ray_paths
        if surface_list[i][0] == 'SPH':
            spherical_surface=surface_list[i][1] #if dealing with 'SPH' then we need to cut out the spherical_surface from surface_list
            refracted_ray_paths[i,:,:]=refraction_2d_sph (incident_rays, spherical_surface) #we input the 2d refracted_rays from refraction_2d_sph into the corresponding section of the 3d refracted_ray_paths
        if surface_list[i][0] == 'DET':
            x_det=surface_list[i][1] #if dealing with 'DET' then we need to cut out x_det from surface_list
            refracted_ray_paths[i,:,:]=refraction_2d_det (incident_rays, x_det) #we input the 2d refracted_rays from refraction_2d_det into the corresponding section of the 3d refracted_ray_paths
	incident_rays=refracted_ray_paths[i,:,:] #ensuring the new incident rays are the refracted_ray_paths from the previous surface
    
    return refracted_ray_paths
    
    
'''task 4'''
import matplotlib.pyplot as pyplot
def plot_trace_2d (incident_rays, refracted_ray_paths, surface_list):
    '''function to plot results of trace_2d
	
    incident_rays consist of a 2d numpy array of floats, the ray is specified by the first index, the second index 
    specifies (in order) horizontal coordinate of ray origin, corresponding vertical coordinate (both in metres) and the clockwise
    angle (in radians) with respect to positive vertical direction of incident ray
	
    refracted ray paths is a 3 dimensional array, of the refracted_rays (from refraction_2d, refraction_2d_sph, refraction_2d_det) for each surface in turn,
    therefore the first index represents the surface and the next two represent the same things as refracted_rays (the return of refraction_2d or refraction_2d_sph
    or refraction_2d_det)
	
    surface_list is a list consisting of seperate lists where the first element is a string saying either 'PLA', 'SPH' or 'DET', 
    this therefore specifies what type of surface is being met, the second element is the corresponding numpy array (or float value) 
    so for PLA this would be planar_surface (as defined in refraction_2d), for SPH this would be spherical_surface (as defined in refraction_2d_sph)
    and for DET this would be x_det (as defined in refraction_2d_det)'''


    import matplotlib.gridspec as gridspec
    gs=gridspec.GridSpec(1,2, width_ratios=[4,1])
    ax1=pyplot.subplot(gs[0])
    ax2=pyplot.subplot(gs[1])    
    
    
    #plotting the very initial ray
    for i in range (len(incident_rays)): #this is to ensure we go through every single ray
        xvalues_initial=(incident_rays[i,0], refracted_ray_paths[0,i,0]) #the xvalues_initial are the initial x values in incident_rays and the first x value in refracted_ray_paths
        yvalues_initial=(incident_rays[i,1], refracted_ray_paths[0,i,1]) #the yvalues_initial are the initial y values in incident_rays and the first y value in refracted_ray_paths
        
        ax1.plot(xvalues_initial, yvalues_initial, "r-o") #plot xvalues_initial against yvalues_initial make the colour of the ray red with end markers
    
    #plotting the surfaces and the rays between
    for i in range(len(surface_list)): #going through all the surfaces 
        if surface_list[i][0] == 'PLA': 
            planar_surface=surface_list[i][1] #cut out the planar_surface from the surface list so we can find equation of surface
            surface_gradient = (planar_surface[3]-planar_surface[1])/(planar_surface[2]-planar_surface[0]) #calculate gradient of planar_surface (as in miniproject)
            surface_intercept = planar_surface[1] - (surface_gradient * planar_surface[0]) #calculate the y intercept of the planar_surface (as in miniproject)
            x=numpy.linspace(planar_surface[0], planar_surface[2],20) #make a linspace of a set of equally spaced xvalues between the two points given
            y=(surface_gradient*x)+surface_intercept #use the equation of the planar_surface line to calculate the corresponding y values
            ax1.plot(x,y,"m") #plot x against y,  make this magenta
            
            for j in range(len(incident_rays)): #go through every single ray
                xvalues=(refracted_ray_paths[i,j,0], refracted_ray_paths[i+1,j,0]) #the xvalues will be between this surface and the next surface in refracted_ray_paths
                yvalues=(refracted_ray_paths[i,j,1], refracted_ray_paths[i+1,j,1]) #the yvalues will be between this surface and the next surface in refracted_ray_paths
                ax1.plot(xvalues, yvalues,"r-o") #plot xvalues against yvalues, make this red to match previous rays, end markers to see where the rays hit the surface
            
        if surface_list[i][0] == 'SPH':
            spherical_surface=surface_list[i][1] #cut out the spherical_surface from the surface_list so we can find equation of the curved surface
            y=numpy.linspace(spherical_surface[1], spherical_surface[3],40) #make a linspace ranging through the yvalues- each y value has one x value- therefore by defining y first we make this a plotable function
            if spherical_surface[6]>0: 
                spherical_centre=spherical_surface[0]+numpy.sqrt(((spherical_surface[6])**2)-((spherical_surface[1])**2)) #if radius is postive (i.e. convex) then centre will be on the right of end points
                x=spherical_centre-numpy.sqrt(((spherical_surface[6])**2)-(y**2))  #if radius is postive (i.e. convex) then all the points will be on the left of the centre
            if spherical_surface[6]<0:
                spherical_centre=spherical_surface[0]-numpy.sqrt(((spherical_surface[6])**2)-((spherical_surface[1])**2)) #if radius is negative (i.e. concave) then centre will be on the left of end points
                x=spherical_centre+numpy.sqrt(((spherical_surface[6])**2)-(y**2)) #if radius is negative (i.e. concave) then all the points will be on the right of the centre
            ax1.plot(x,y,"g") #plot x against y for the spherical surface, this is made green to match planar surface
            
            for j in range(len(incident_rays)): #go through every single ray
                xvalues=(refracted_ray_paths[i,j,0], refracted_ray_paths[i+1,j,0]) #the xvalues will be between this surface and the next surface in refracted_ray_paths
                yvalues=(refracted_ray_paths[i,j,1], refracted_ray_paths[i+1,j,1]) #the yvalues will be between this surface and the next surface in refracted_ray_paths
                ax1.plot(xvalues, yvalues,"r-o") #plot xvalues against yvalues, make this red to match previous rays, end markers to see where the rays hit the surface
             
        if surface_list[i][0] == 'DET':
            x_det=surface_list[i][1] #cut out the value of x_det from the surface_list
            ax1.axvline(x=x_det) #plot as a simply x line, as cannot plot x against y for this
        
        #detectors plot
        if surface_list[i][0] == 'DET':
            x_det=surface_list[i][1] #cut out the value of x_det from the surface_list
            for j in range(len(incident_rays)): #go through every single ray
                xvalue= x_det#the xvalues will be between this surface and the next surface in refracted_ray_paths
                yvalue=refracted_ray_paths[i,j,1] #the yvalues will be between this surface and the next surface in refracted_ray_paths
                ax2.plot(xvalue, yvalue,"ro") #plot xvalues against yvalues, make this red to match previous rays, end markers to see where the rays hit the surface
            
    ax1.set_xlabel('x position/m') #create a label for x axis
    ax1.set_ylabel('y position/m') #create a label for y axis
    ax1.set_title ('plot trace 2d') #create a title
    ax2.set_xlabel('x position/m') #create a label for x axis
    ax2.set_ylabel('y position/m') #create a label for y axis
    ax2.set_title ('1D image on the det') #create a title
    pyplot.show()


'''task 5'''

def evaluate_trace_2d (refracted_ray_paths, r):
    '''function to determine the fraction of rays which fall within a circle of radius r (on the deector), centred at the mean position of the 
    rays at the detector
    
    refracted ray paths is a 3 dimensional array, of the refracted_rays (from refraction_2d, refraction_2d_sph, refraction_2d_det) for each surface in turn,
    therefore the first index represents the surface and the next two represent the same things as refracted_rays (the return of refraction_2d or refraction_2d_sph
    or refraction_2d_det)
    
    r is a float value specifying the radius of the circle (in metres)
    
    frac is a float value specifying the fraction of rays within the circle'''
    
    # The number of the rays we are dealing with
    k=(len(refracted_ray_paths))-1 #need index of last surface, therefore amount of surfaces minus one
    y_values= refracted_ray_paths[k,:,1]  #don't need x values because all have same x values det surface
    centre_of_circle=sum(y_values)/float(len(y_values)) #finding average by summing and dividing by no. of rays
    b=0 #starting the count of no of rays within the circle at zero
    for i in range(len(y_values)):
        if abs(refracted_ray_paths[k,i,1]-centre_of_circle)<=r:
            b=b+1 #if the y value is within r metres away from the centre the count will go up
        if abs(refracted_ray_paths[k,i,1]-centre_of_circle)>r:
            b=b # if the y value is not within r metres away from the centre the count will not go up
    frac=b/float(len(y_values)) #the ratio of rays within the circle is the count of rays in the circle over the amount of rays
    return frac
    
'''task6'''
from scipy.optimize import minimize 

def optimize_surf_rad_2d (incident_rays,surface_list, r, n_surf):
    '''a function to optimize the radius of curvuture to give the highest value of frac (defined within evaluate_trace_2d)
    incident_rays consist of a 2d numpy array of floats, the ray is specified by the first index, the second index 
    specifies (in order) horizontal coordinate of ray origin, corresponding vertical coordinate (both in metres) and the clockwise
    angle (in radians) with respect to positive vertical direction of incident ray
	
    surface_list is a list consisting of seperate lists where the first element is a string saying either 'PLA', 'SPH' or 'DET', 
    this therefore specifies what type of surface is being met, the second element is the corresponding numpy array (or float value) 
    so for PLA this would be planar_surface (as defined in refraction_2d), for SPH this would be spherical_surface (as defined in refraction_2d_sph)
    and for DET this would be x_det (as defined in refraction_2d_det)
	
    r is a float value specifying the radius of the circle where the rays need to meet the detector within (in metres)
    
    n_surf is an array specifying the index of the surfaces in surface_list which are spherical surfaces which need to be optimised
    
    The returned function is an array of the optimized radius' '''
    
    #define a function that runs evaluate_trace_2d in terms of an array of all the radius of curvutures of all the spherical surfaces (defined as 'R')
    def function (R):
        for i in range(len(R)): #going through every index of a radius in the array R
            surface_list[n_surf[i]][1][6] = R[i] #inputting the values of R into the surface_list using the indexes from n_surf
        refracted_ray_paths=trace_2d (incident_rays, surface_list) #finding the refracted_ray_paths (defined using the new surface list)
        frac=evaluate_trace_2d(refracted_ray_paths, r) #find frac as defined by evaluate_trace_2d
        frac_new=-1.0*frac #define frac_new as negative of the frac (so this can be minimised to maximise frac)
        return frac_new
    
    R=numpy.zeros(len(n_surf),dtype=float) #create an array to hold radius values
    for i in range(len(n_surf)):
        R[i]=surface_list[n_surf[i]][1][6] #going through all the surface_list incidicies within n_surf, to find all the radius of curvuture
    
    res = minimize(function, R, method= 'nelder-mead', tol=1e-6) #using the minimise function using the nelder-mead method optimise the 'function' which obtains frac_new in terms of the radii
    
    #this will result in a return of the values of the radii which would give a maximum for frac (minimum for frac_new)
    return res.x



if __name__ == "__main__":
    surface_list=(['SPH', numpy.array([8,9,8,-9,1,1.5,50])], ['SPH', numpy.array([8,9,8,-9,1.5,1,-50])],['PLA', numpy.array([25,-10,28,10.0,1.0,1.33])], ['DET', 55] )
    incident_rays=numpy.zeros((30,3))
    incident_rays[:,1]=numpy.linspace(-8,8,30)
    incident_rays[:,2]=numpy.pi/2.0
    incident_rays[15,1]=0.1
    refracted_ray_paths=trace_2d(incident_rays, surface_list)
    print refracted_ray_paths
    plot_trace_2d (incident_rays, refracted_ray_paths, surface_list)
    print evaluate_trace_2d (refracted_ray_paths, 1)
    n_surf=numpy.array([0,1])
    print optimize_surf_rad_2d (incident_rays, surface_list, 1, n_surf)
	
    
    
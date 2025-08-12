from fenics import *
import numpy as np
import sys
import os

project = os.environ["PROJECT"]
if sys.argv[2]=='uul':
    case=sys.argv[1]+'_uul'
    folder = 'case_{}'.format(sys.argv[1])
    if sys.argv[1]=='A':
        case_names = ['PTSeg028_uul_0p8','PTSeg028_uul_0p64','PTSeg028_uul_0p512','PTSeg028_uul_0p4096']
    elif sys.argv[1]=='B':
        case_names = ['PTSeg043_uul_0p8','PTSeg043_uul_0p64','PTSeg043_uul_0p512','PTSeg043_uul_0p4096']#
    elif sys.argv[1]=='C':
        case_names = ['PTSeg106_uul_0p8','PTSeg106_uul_0p64','PTSeg106_uul_0p512','PTSeg106_uul_0p4096']
    case_names = [name for name in case_names if '4096' in name]
elif sys.argv[2] =='base':
    case=sys.argv[1]+'_base'
    folder = 'case_{}'.format(sys.argv[1])
    if sys.argv[1]=='A':
        case_names = ['PTSeg028_base_0p8','PTSeg028_base_0p64','PTSeg028_base_0p512','PTSeg028_base_0p4096']
    elif sys.argv[1]=='B':
        case_names = ['PTSeg043_base_0p8','PTSeg043_base_0p64','PTSeg043_base_0p512','PTSeg043_base_0p4096']#
    elif sys.argv[1]=='C':
        case_names = ['PTSeg106_base_0p8','PTSeg106_base_0p64','PTSeg106_base_0p512','PTSeg106_base_0p4096']
    #case_names = [name for name in case_names if '4096' in name]
else:
    if sys.argv[1]=='Groccia':
        case='Groccia'
        folder=project+'/Swirl/swirl_cases'
        case_names=['Groccia']
    else:
        case = sys.argv[1]
        folder = project+'/mesh_rez/data/cases/case_{}'.format(sys.argv[1])
        case_names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

for case_name in case_names:
    # Create mesh and define function space
    mesh_folder = folder+'/'+ case_name + '/data'
    if sys.argv[2]=='notref':
        splits = case_name.split('_')
        seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]#splits[-2]
    else:
        seg_name = case_name
    #seg_name = 'Groccia'
    mesh_file = mesh_folder +'/' + seg_name + '.xml.gz'
    mesh = Mesh(mesh_file)
    fd = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
    '''
    dsw = ds(1, domain=mesh, subdomain_data=fd) #walls
    dsss = ds(2, domain=mesh, subdomain_data=fd) #sup sag
    if case == 'B':
        dsl = ds(3, domain=mesh, subdomain_data=fd) #labbe
        dso = ds(4, domain=mesh, subdomain_data=fd) #outlet
    else:
        dso = ds(3, domain=mesh, subdomain_data=fd) #outlet
    '''

    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define boundary conditions
    #zero gradient on walls
    bcs = []
    bcw = DirichletBC(V, Constant(0.0), fd, 0)
    if case == 'B' or case=='B_uul' or case=='B_base':
        bcl = DirichletBC(V, Constant(80.0), fd, 2)
        bco = DirichletBC(V, Constant(0.0), fd, 3)
        bcs.append(bcl)
        bcs.append(bco)
        bcss = DirichletBC(V, Constant(80.0), fd, 1)
        bcs.append(bcss)
    elif case == 'Groccia':
        bcl = DirichletBC(V, Constant(100.0), fd, 2)
        bcsy = DirichletBC(V, Constant(100.0), fd, 3)
        bct = DirichletBC(V, Constant(80.0), fd, 4)
        bco = DirichletBC(V, Constant(80.0), fd, 5)
        bcss = DirichletBC(V, Constant(0.0), fd, 1)
        bcs.append(bcss)
        bcs.append(bco)
        bcs.append(bcl)
        bcs.append(bcsy)
        bcs.append(bct)
    else:
        bco = DirichletBC(V, Constant(0.0), fd, 2)
        bcs.append(bco)
        bcss = DirichletBC(V, Constant(80.0), fd, 1)
        bcs.append(bcss)
    '''
    bc_w_len = len(bcw.get_boundary_values())
    print( 'Wall BC on ' + str(bc_w_len) , 'cells')
    bc_ss_len = len(bcss.get_boundary_values())
    print( 'SS BC on ' + str(bc_ss_len) , 'cells')
    bc_o_len = len(bco.get_boundary_values())
    print( 'Outlet BC on ' + str(bc_o_len) , 'cells')

    '''
    k = Constant(1)
    f = Constant(0)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = k*inner(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V, name = 'T')
    solve(a == L, u, bcs)

    vtkfile = File('case_{}/{}_heat.pvd'.format(sys.argv[1], case_name))
    vtkfile << u

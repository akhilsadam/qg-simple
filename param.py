# import math
import numpy as np
# class param:
    
#     project_name = 'qg' # usually set in config
#     # runner = 'run.py' # usually set in tasks
    
#     class grid:
#         Nx = 512
#         Ny = 512
#         Lx = 2 * math.pi
#         Ly = 2 * math.pi
        
#     class time:
#         dt = 1e-3
#         T = 100
#         save_rate = 50        # (Frequency of .np saves and plots)
        
#     class pde:
#         mu = 0.0               # (Linear drag)
#         nu = 1.025e-5          # (Viscosity coefficient)
#         B = 0.0                # (Beta plane)
#         nv = 1                 # (Hyperviscous order)
#         penalty = 1.25         # (Brinkman penalty parameter)
#         friction = 0.0         # (friction coefficient - between no-slip and free-slip)
#         # rossby_radius = 1.0   # (Rossby radius for vortex stretching) TODO Fix
        
#     class ic:
#         energy = 0.0000001
#         wavenumbers = [10.0, 32.0]
#         seed= 86   

#     class forcing:
#         function = 'none'

#     class bc:
#         function = 'inlet-outlet'
#         inlet_velocity = 0.2
#         mn = 0.0
#         mx = 1.0
#         eta = 1.25
#         inlet_x = 0.10

#     class mask:
#         function = 'image' #'circular'
#         mask = 'osk.png'
#         # r = math.pi/4
#         # tol = 1e-3


res = 128
config = type('config', (object,), {
    'actions':[ \
        type('action', (object,), { 
            'action_name':'convergence_test',
            'no_compute':False,
            'strict_version_checks':False,
            'base_save_path':'../run/',
            'debug':False,
            'save_frequency':100,
            'ngpu':1,
            'data':'',
            'tasks':[ \
                type('task', (object,), { 
                    'task_name':name,
                    'runs':[ \
                        type('validate', (object,), { 
                            'runner':'run.py',
                        'grid':type('grid', (object,), { 
                                'Nx':res,
                                'Ny':res,
                                'Lx':6.283185307179586,
                                'Ly':6.283185307179586,
                                'precision':'float32',
                             }),
                        'time':type('time', (object,), { 
                                'dt':dt,
                                'T':200,
                                'save_rate':1000,
                             }),
                        'pde':type('pde', (object,), { 
                                'mu':0.0,
                                'nu':1.025e-3,
                                'B':0.0,
                                'nv':1,
                                'penalty':penalty,
                                'friction':0.0,
                                'rossby_radius':None,
                             }),
                        'ic':type('ic', (object,), { 
                                'function':'randn',
                                'energy':0.01,
                                'wavenumbers':[ \
                                    10.0,
                                    32.0,
                                ],
                                'seed': 86,
                             }),
                        'bc': None,
                        'forcing': None,
                        # 'forcing':type('forcing', (object,), {
                        #         'function':'unscaled_cosine',
                        #         'A':0.25,
                        #         'B':4,
                        #         'C':0,
                        #         'D':0.25,
                        #         'E':4,
                        #         'F':0,
                        # }),
                        # 'mask':type('mask', (object,), { 
                        #         'function':'image',
                        #         'r':0.7853981633974483,
                        #         'tol':0.001,
                        #         'mask':'osk.png',
                        #      }),
                            'project_name':'qg',
                         }) for dt in [0.01,0.002,0.001,0.0005,0.00025] # 64, 80, 100, 128, 160,
                    ],
                 }) for name,penalty in [('pure', 0.0)] #, ('pure', 0.0)
            ],

         }),
    ],
    'project_name':'qg',
    'cluster_name':'mseas.mit.edu',
 })

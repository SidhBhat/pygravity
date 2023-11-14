import gravity
# import pdb

env = gravity.Environment();

# for i in range (-5,5,1):
# 	env.add_body(mass=1,position=[i,i]);

env.add_body(name='earth',color='b',mass=10,position=[0,0],velocity=[0.181,0]);
env.add_body(name='moon',color='k',mass=1,position=[0,3],velocity=[-1.769,0]);
env.add_body(name='satelite',color='r',mass=0.1,position=[0,3.5],velocity=[-0.355,0]);
#
# env.add_body(name='body 1',color='b',mass=100,position=[0,0],velocity=[0,0]);
# env.add_body(name='body 2',color='r',mass=100,position=[-2,0],velocity=[0,7]);
# env.add_body(name='body 3',color='g',mass=50,position=[4,0],velocity=[-7,0.1]);

# env.add_body(name='black hole',color='k',mass=100000,position=[0,0],velocity=[0,0]);
# env.add_body(name='satelite',color='r',mass=10,position=[10,0],velocity=[0,80]);
# env.add_body(name='satelite',color='r',mass=1,position=[-10,0],velocity=[0,95]);
#
# # env.set_markersize(100);
env.set_fps(60);
env.set_markersize(50);
env.set_timestep(0.00005);
env.set_bounds(5);

env.simulate(100);

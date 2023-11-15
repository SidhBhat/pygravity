import gravity

# create Environment
env = gravity.Environment();

env.add_body(name='earth',color='b',mass=5.972168e24,position=[0,0],velocity=[0,0]);
env.add_body(name='moon',color='k',mass=7.342e22,position=[3.626e8,0],velocity=[0,1.1e3]);
env.set_gravitational_constant(6.6743e-11)

env.set_timestep(0.01);
env.set_bounds(4.5e8);
env.set_markersize(1e12)

env.simulate(100);
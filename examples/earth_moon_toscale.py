import gravity

# create Environment
env = gravity.Environment();

# The earth Moon System (as per data available on wikipedia)
earth = env.add_body(name='earth',color='b',mass=5.972168e24,position=[0,0],velocity=[0,0]);
moon  = env.add_body(name='moon',color='k',mass=7.342e22,position=[3.626e8,0],velocity=[0,1.1e3]);
env.set_gravitational_constant(6.6743e-11)
earth.set_parameters(radius=6.371e6);
moon.set_parameters(radius=1.737e6);

env.set_timestep(0.01);
env.set_bounds(4e6);
env.set_origin(moon['position']);

env.draw();

env.simulate(100, plot_simulation_result=True);


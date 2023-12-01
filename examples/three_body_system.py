import gravity

env = gravity.Environment();

# Three bodies orbiting each other
sun   = env.add_body(name='Sun',color='y',mass=10,position=[0,0],velocity=[0.181,0]);
earth = env.add_body(name='earth',color='b',mass=1,position=[0,3],velocity=[-1.769,0]);
moon  = env.add_body(name='moon',color='k',mass=0.1,position=[0,3.5],velocity=[-0.355,0]);

## veiw from COM frame of referance

env.set_origin(sun['position']);
sun['radius'] = 0.7;
# plot propertis
env.set_radius(0.05);
env.set_timestep(0.0005);
env.set_bounds(10);

env.show()
env.simulate(10, plot_simulation_result=True);

## view from the moon frame of referance
env.set_origin(moon['position']);
env.lock_frame_on(moon);

env.show()
env.simulate(10);

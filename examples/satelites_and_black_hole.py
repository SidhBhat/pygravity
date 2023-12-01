import gravity

env = gravity.Environment();

# two small bodies oribiting a massive body
env.add_body(name='black hole',color='k',mass=100000,radius=2,position=[0,0],velocity=[0,0]);
env.add_body(name='satelite',color='r',mass=10,position=[9,0],velocity=[0,105.5]);
env.add_body(name='satelite',color='b',mass=10,position=[-10,0],velocity=[0,100]);

# plot propertis
env.set_fps(100);
env.set_bounds(24);
env.set_timestep(0.00001);
env.get_body("black hole")['radius'] = 1;

env.simulate(10);
